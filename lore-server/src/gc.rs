// SPDX-FileCopyrightText: 2026 Epic Games, Inc.
// SPDX-License-Identifier: MIT
//! Reachability-based garbage collection for a server's immutable store.
//!
//! The store's own [`ImmutableStore::evict`] ranks by last access and capacity, which
//! is right for a client cache that can re-fetch, and wrong for a server where the
//! store is the source of truth: it drops fragments a live repository still needs.
//! Compaction is reference-aware but only ever measures and rewrites the packstore,
//! so it cannot reclaim what accumulates outside it.
//!
//! This walks every repository, branch and revision the server holds, collects the
//! addresses they reference, and sweeps the rest.
//!
//! **The mark must be complete or the sweep destroys data.** [`run`] therefore
//! defaults to a dry run, and ships with a self-check: on a store whose repositories
//! are all live, a complete mark leaves nothing collectable, so a non-zero count is
//! proof the walk missed addresses and that deletion must not be enabled.
use std::collections::HashSet;
use std::sync::Arc;

use lore_base::types::Address;
use lore_base::types::Context;
use lore_revision::branch;
use lore_revision::repository;
use lore_revision::repository::RepositoryContext;
use lore_revision::state;
use lore_revision::state::State;
use lore_storage::ImmutableStore;
use lore_storage::MutableStore;
use lore_storage::store_types::StoreObliterateStats;
use tokio_stream::StreamExt;

/// What a pass looked at and what it would remove.
#[derive(Debug, Default, Clone, Copy)]
pub struct GcReport {
    pub repositories: usize,
    pub branches: usize,
    pub revisions: usize,
    pub live_addresses: usize,
    pub scanned: usize,
    pub collected: usize,
    /// Unreferenced but inside the grace window, so deliberately left this pass.
    pub protected: usize,
    pub dry_run: bool,
}

/// Mark `address` and, when its payload is fragmented, the sub-fragments holding
/// its bytes. Marking the top-level address alone leaves those sweepable, which
/// is a repository that loads until the first read of its metadata.
async fn mark_expanded(
    repo: &Arc<RepositoryContext>,
    address: Address,
    live: &mut HashSet<Address>,
) -> Result<(), String> {
    match state::collect_new_addresses(repo.clone(), &[address], false).await {
        Ok(expanded) => {
            live.extend(expanded);
            live.insert(address);
            Ok(())
        }
        Err(err) => Err(format!("failed to expand {address}: {err}")),
    }
}

/// Collect every address reachable from any repository, branch and revision.
///
/// Collecting against a default (empty) parent state asks for the whole set a
/// revision names rather than the delta against its parent, which is what the push
/// path uses the same call for.
pub async fn collect_live_addresses(
    immutable_store: Arc<dyn ImmutableStore>,
    mutable_store: Arc<dyn MutableStore>,
    report: &mut GcReport,
) -> Result<HashSet<Address>, String> {
    let root = Arc::new(RepositoryContext::new_server_context(
        immutable_store,
        mutable_store,
        Context::default().into(),
    ));

    let mut live: HashSet<Address> = HashSet::new();

    let mut repositories = repository::list_local(root.clone())
        .await
        .map_err(|err| format!("failed to list repositories: {err}"))?;

    while let Some(id) = repositories.next().await {
        // Deleting a repository leaves a zero id in the listing. It is not a
        // repository and owns no fragments, so skipping it is safe — whereas
        // failing the walk on it would stop collection entirely after any delete.
        if id.is_zero() {
            lore_base::lore_debug!("GC mark: skipping zero repository id (deleted repository)");
            continue;
        }
        report.repositories += 1;
        let repo = Arc::new(root.to_server_context(id.into()));

        // The repository's own metadata fragment is reachable but named by neither a
        // branch nor a revision, so nothing above would have marked it.
        match repository::metadata_hash(repo.clone()).await {
            Ok(hash) if !hash.is_zero() => {
                lore_base::lore_debug!("GC mark: repository {id} metadata {hash}");
                mark_expanded(&repo, Address::zero_context_hash(hash), &mut live).await?;
            }
            Ok(_) => lore_base::lore_debug!("GC mark: repository {id} has no metadata hash"),
            Err(err) => {
                return Err(format!("failed to load metadata hash for {id}: {err}"));
            }
        }

        let mut branches = match branch::list(repo.clone()).await {
            Ok(branches) => branches,
            Err(err) => {
                lore_base::lore_warn!("GC: failed to list branches for {id}: {err}");
                continue;
            }
        };

        while let Some(branch_id) = branches.next().await {
            report.branches += 1;

            // Likewise a branch's metadata fragment: the revision walk names the
            // revisions a branch points at, never the branch record itself.
            match branch::metadata_hash(repo.clone(), branch_id.into()).await {
                Ok(hash) if !hash.is_zero() => {
                    lore_base::lore_debug!("GC mark: branch {branch_id} metadata {hash}");
                    mark_expanded(&repo, Address::zero_context_hash(hash), &mut live).await?;
                }
                Ok(_) => lore_base::lore_debug!("GC mark: branch {branch_id} has no metadata hash"),
                Err(err) => {
                    return Err(format!("failed to load branch metadata for {branch_id}: {err}"));
                }
            }

            let revisions =
                match branch::list_revisions(repo.clone(), Some(branch_id), None, None, None).await
                {
                    Ok(revisions) => revisions,
                    Err(err) => {
                        lore_base::lore_warn!("GC: failed to list revisions for {branch_id}: {err}");
                        continue;
                    }
                };

            for item in revisions.revisions.iter() {
                report.revisions += 1;
                let revision = item.revision;

                // The revision record itself, and the parents it names. Collecting
                // against a default from-state does not reliably yield these, and a
                // revision whose own record is swept is a revision that cannot load.
                live.insert(Address::zero_context_hash(revision));
                if !item.parent_self.is_zero() {
                    live.insert(Address::zero_context_hash(item.parent_self));
                }
                if !item.parent_other.is_zero() {
                    live.insert(Address::zero_context_hash(item.parent_other));
                }
                let state = match State::deserialize(repo.clone(), revision).await {
                    Ok(state) => state,
                    Err(err) => {
                        // A revision we cannot read is a revision we cannot prove the
                        // reachability of, so the pass must not delete anything.
                        return Err(format!("failed to load state for {revision}: {err}"));
                    }
                };

                let empty = Arc::new(State::default());
                match state::collect_new_fragments(repo.clone(), empty, state, false).await {
                    Ok(addresses) => {
                        lore_base::lore_debug!(
                            "GC mark: revision {revision} contributed {} addresses",
                            addresses.len()
                        );
                        live.extend(addresses);
                    }
                    Err(err) => {
                        return Err(format!("failed to collect fragments for {revision}: {err}"));
                    }
                }
            }
        }
    }

    report.live_addresses = live.len();
    Ok(live)
}

/// Run one pass. `dry_run` counts what would go and changes nothing.
pub async fn run(
    immutable_store: Arc<dyn ImmutableStore>,
    mutable_store: Arc<dyn MutableStore>,
    dry_run: bool,
    grace_seconds: u64,
) -> Result<GcReport, String> {
    let mut report = GcReport {
        dry_run,
        ..Default::default()
    };

    let live = collect_live_addresses(
        immutable_store.clone(),
        mutable_store.clone(),
        &mut report,
    )
    .await?;

    let stats = Arc::new(StoreObliterateStats::default());
    let sweep = immutable_store
        .sweep_unreferenced(&live, dry_run, grace_seconds, stats)
        .await
        .map_err(|err| format!("sweep failed: {err}"))?;

    report.scanned = sweep.scanned;
    report.collected = sweep.collected;
    report.protected = sweep.protected;
    Ok(report)
}
