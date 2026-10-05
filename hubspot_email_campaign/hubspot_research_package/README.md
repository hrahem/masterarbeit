# HubSpot thesis dataset: read this first

This package contains pseudonymized CSVs, a combined dataset, extracted email features, and a data-quality audit. Original uploads were not changed. Start with `AUDIT_REPORT.html`, then `primary_pilot_7d.csv`.

## What to open

| File | Purpose |
|---|---|
| `AUDIT_REPORT.html` | Standalone readable findings, counts, and next steps. Open in a browser. |
| `pseudonymized_events/<source folder>/events.csv` | A sanitized event export for each of the 10 supplied folders, including the flagged duplicate for provenance. |
| `combined_events.csv` | Nine distinct campaigns, deduplicated across folders. Superseded events remain here and are explicitly marked. |
| `recipient_send_table.csv` | One row per message with an active SENT or DELIVERED event; includes delivery failures and immature outcomes, with flags. |
| `primary_pilot_7d.csv` | Conservative pilot subset: delivered, seven-day mature, test phase, bot filtering reported on, and no likely internal recipient. |
| `campaign_audit.csv` | Source-derived counts versus the performance screenshots, including discrepancies. |
| `pilot_ab_baseline.csv` | Same-window descriptive A/B results for the primary pilot, without pooled winner claims. |
| `variant_summary.csv` | Descriptive results by campaign, variant, and phase. All-observed and seven-day outcomes are separate. |
| `email_features.csv` | Subject templates, structural HTML counts, and content-family grouping. |
| `quality_issues.csv` | Specific limitations and follow-up actions. |
| `source_manifest.csv` | Original-folder mapping, source CSV hashes, and duplicate handling. |
| `summary.json` | Machine-readable headline statistics and validation status. |
| `build_campaign_package.py` | The transformation code used for the CSVs; standard Python library only. |

## Privacy transformations

- Known recipient addresses are replaced with HMAC-based opaque IDs. The same normalized address gets the same ID across this batch. Addresses are trimmed and lowercased; aliases are not merged.
- Hidden/missing recipients receive message-scoped IDs. These do NOT identify the same person across sends. Never group all hidden contacts together.
- Raw message IDs, event IDs, and related-event references are pseudonymized. URLs become opaque link IDs with no exposed domain, path, or query string.
- Personalized subject lines in the September campaign are replaced by `[RECIPIENT], heute endet dein Preisvorteil.` Names and addresses embedded in subjects are removed.
- Sender/reply-to/CC/BCC details, user-agent strings, city/state/country, raw bounce explanations, free-text messages, and other unneeded source fields are omitted through an explicit field allowlist.
- Campaign names, source campaign IDs, timestamps, public subject templates, and engagement records are retained for research. This is pseudonymization, NOT a guarantee of irreversible anonymization. Combining these records with the original company data can allow linkage.
- Original screenshots and HTML files are intentionally not included in this research package: they can contain names, email addresses, and personalized links. Original files remain in the source upload.
- No original-to-pseudonym lookup table or secret key is included. The generated key was held in memory only. IDs are consistent inside this release. To add later campaigns, rebuild the entire combined batch, or run the script yourself with a securely managed `HUBSPOT_PSEUDONYM_KEY` consistently across runs. Do not put a real key in the script, package, or thesis. New runs without the same key produce different IDs but the same analytical counts for identical inputs.

## Cleaning and outcome definitions

The folders `dach_de_plus_m_consistency26m` and `dach_de_plus_m_consistency26m_3` contain the same 500 event records. The `_3` folder is canonical. Both sanitized source copies are included, but only one enters the combined data. Do not concatenate the per-folder exports again.

Event identity is checked using source portal and event ID. A message is grouped using source portal, campaign ID, and message ID. Repeated clicks are counted once for a binary outcome, while event counts remain available separately. Superseded events are retained in the event export but excluded from the send/outcome calculations.

The cutoff is **2026-09-21 00:00:00 in the source export's local clock**, a conservative start-of-export-day boundary. The export says "Your time zone" without naming the zone. Timestamps have NOT been converted to UTC or guessed to be Europe/Berlin. Confirm the account time zone before clock-time or delay-sensitive deployment analysis.

`clicked_7d` means at least one active CLICK event from send time through seven days after send, on a delivered message with a complete window. Recent sends and undelivered messages have a blank label, not zero. All labels for the September 18 campaign remain blank because its seven-day window is incomplete. Re-export on September 26 or later. Do not treat its observed clicks as final seven-day outcomes.

All-observed counts use the entire supplied export, including late clicks and records after the conservative cutoff, to reconcile with the screenshots. These counts must not be substituted for fixed-window training labels. Observed clicks are not automatically conversions or guaranteed human clicks. Historical export completeness remains a source assumption.

`primary_pilot_7d.csv` is a starting dataset, not a validated causal benchmark. It excludes the 2023 unfiltered campaign, the immature September campaign, winner/fallback rollouts, and likely internal recipients whose addresses use `provenexpert.com` or `bewertet.de`. The internal classification is a heuristic for company review; full pseudonymized data retains these records with a flag. These exclusions were chosen before fitting any model.

Reported bot filtering is a campaign-level screenshot setting. There is no per-event bot label in these exports. Open counts differ slightly in several campaigns even after superseded records are removed. These discrepancies are disclosed rather than patched with invented rows. Delivery and unique-click counts reconcile across all nine distinct campaigns. Active SENT counts still differ for the 2023 and September 2026 campaigns.

## Features and leakage prevention

- `prior_delivered_in_exports` counts distinct earlier messages whose active delivery event occurred before the current send.
- `prior_clicked_messages_in_exports` counts distinct earlier messages with a first active click before the current send.
- `days_since_prior_click_in_exports` uses the most recent such click, not the eventual outcome of earlier campaigns.
- These histories only cover the supplied campaigns. Zero means no prior event observed in these exports; it does not mean an inactive or new customer. Hidden identities have blank histories.
- Current-message outcomes, phase, event timestamps, first-click delay, and delivery status are audit/outcome fields, not pre-send predictor inputs. Subject template, design features, campaign context known at sending, and pre-send history are candidate inputs.
- The CW12 2025 newsletters sent to different audiences share a `content_family`. Keep that family together for evaluation of generalization to new content. A and B are local labels; A in one campaign is not the same action as A in another. The A/B subject ordering is even reversed for one audience in this batch.
- Image/link element counts include footer, tracking, and hidden markup. They are rough HTML structure features, not verified counts of visible images or CTAs.
- Matching extracted body text does not prove identical visual design. Render or inspect HTML/CSS before making causal claims about layout or color.

## Assignment and uncertainty

Test/rollout mapping uses subjects, campaign IDs, send timing, and screenshots. Screenshots report either 50/50 tests or 25/25/50 test-and-rollout configurations. `configured_p_A_given_test=0.5` is conditional on membership in the test phase, not a verified per-recipient propensity for the whole audience. `assignment_mechanism_verified=0` is intentional. Do not use a constant 0.5 for rollout records or claim unbiased IPS/replay evaluation without verifying assignment and eligibility assumptions.

HubSpot's reported winner used opens per delivery in the staged tests. It is not a demonstrated click-rate winner. The reported 2026 winners have not been independently retested here.

Wilson intervals in `variant_summary.csv` describe binomial uncertainty around all-observed per-message rates. They do not adjust for repeat recipients, multiple comparisons, selection, audience differences, bot activity, or shared campaign content. They are not proof of a causal winner. There is no pooled A-versus-B winner across unrelated campaigns.

## Suggested next implementation

1. Review the duplicate and small residual discrepancies in `quality_issues.csv`. Confirm time zone, randomization, and internal-address exclusions with the company.
2. Start with descriptive A/B click results and simple Beta-Bernoulli Thompson Sampling simulations at realistic campaign sizes. Simulated outcomes must be labeled as such.
3. Use campaign/content-group holdouts and chronological validation; never randomly split rows and claim unseen-campaign generalization.
4. Compare a constant click predictor with a small regularized model only after defining a defensible split. There are few independent content groups and relatively few click outcomes. Do not start with deep learning or a large text model.
5. Collect more recent varied campaigns and consented stable pseudonymous CRM identifiers/context. Re-export September 18 after its observation window matures.
6. Keep drip-campaign RL separate until actual linked journey steps, available actions, and follow-up outcomes are available. Two similarly named sends alone do not establish a complete RL environment.

## Rebuilding

Extract the original uploaded ZIP into a private directory, then run:

```text
python build_campaign_package.py /path/to/hubspot_email_campaign /path/to/output
```

The script is specific to this audited batch, including manually transcribed screenshot metadata and known variant mappings. New campaigns require reviewing and updating that metadata. It intentionally stops if its assumptions fail. It produces the CSVs and summary JSON; this README and HTML report are the accompanying reviewed explanation.

## Field guide

| Fields | Meaning |
|---|---|
| `campaign_id`, `campaign_name` | Canonical batch campaign key and original folder name. |
| `source_campaign_id` | HubSpot email/variant/send-phase ID, retained for campaign-level audit. |
| `variant`, `phase` | A/B design and `test`, `rollout_winner`, or `rollout_fallback`. |
| `recipient_id`, `recipient_linkable` | Pseudonym and whether the original address permitted cross-send linkage. |
| `internal_domain_candidate` | Heuristic probable staff/test domain flag. |
| `message_id`, `event_id` | Pseudonymized keys for sends and individual events. |
| `event_type`, `event_at_local` | Exported event category and local event timestamp. |
| `sent_at_local` | Local send timestamp retained from the source. |
| `event_superseded`, `superseded_by_event_id`, `caused_by_event_id` | Exported event lineage; later replacement marks an event inactive for analysis. |
| `link_id` | Opaque ID derived from the full source click URL; query variations may produce different IDs. |
| `not_sent_reason` | Uppercase source reason code; other free text is redacted. |
| `subject_template` | Safe content template; no recipient-specific name. |
| `bot_filtering_reported` | Screenshot-level `on` or `unavailable`; not a per-event classification. |
| `sent_event_active`, `delivered` | Whether a nonsuperseded event of that type exists for the message. |
| `opened_observed`, `clicked_observed`, `click_events_observed` | Outcomes across the full supplied export, not a fixed observation window. |
| `first_click_delay_hours` | Delay from send to first valid observed click; blank if none. |
| `window_7d_complete`, `clicked_7d`, `analysis_cutoff_local` | Fixed-window completeness, eligible binary target, and conservative observation boundary. |
| `configured_p_A_given_test`, `assignment_mechanism_verified` | Reported test allocation and explicit lack of independent assignment verification. |
| `content_family` | Related creative grouping for validation. |
| `eligible_primary_pilot` | Combined flag for the conservative subset described above. |
| `prior_*_in_exports`, `days_since_prior_click_in_exports` | Limited-history pre-send features defined above. |
| `dashboard_*`, `*_delta` | Manually transcribed performance counts and active-export minus dashboard differences. |
| `test_duration_hours`, `test_outcome`, `rollout_variant`, `winning_metric` | Screenshot settings; duration zero means even split with no fixed stopping duration specified. |
| `test_split_A`, `test_split_B`, `rollout_share` | Configured fractions of the overall audience. |
| `clicked_messages`, `opened_messages`, `delivered` in audit | Distinct messages with active qualifying events; raw event counts are separate. |
| `mature_delivered_7d`, `mature_7d_delivered` | Delivered messages with a complete seven-day observation window. |
| `click_rate_observed`, `wilson95_*` | All-observed descriptive proportions and unadjusted intervals; see limitations. |
| `subject_*`, `html_*`, `body_text_*` | Safe template and extracted structural characteristics; text hash is not a rendering comparison. |
| `included_in_combined`, `duplicate_of`, `source_csv_sha256` | Provenance and file-level duplicate handling. |
| `severity`, `issue`, `action` | Data-quality finding and recommended response. |
