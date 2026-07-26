# Freeze audit — Q-R1 matched-retention factorial v4

**Verdict: `FREEZE_AUDIT_PASS`.** The freeze is valid and the development sequence is
authorized: the `static-bar` step first, then the development workers. The confirmation block
stays sealed.

Audited at `86dc15b4` ("Freeze Q-R1 matched-retention factorial v4") on `codex/q-r1-oracle-v2`,
with `74f8d20d` and `f7975524` as its parents. Every value below was recomputed here.

## The property that mattered most: the frozen contract is what I reviewed

`contracts/q_r1_matched_retention_factorial_v4.json` at `86dc15b4` hashes to

```
bb92a2cbfcd3691a77f7f9ab8a269d7ffab65823d37b41f70d0b13795d92e764
```

which is byte-for-byte the DRAFT covered by `PASS_PRE_FREEZE`. I said a single differing byte
would void that verdict; none differ, so the PASS applies to the frozen document.

The receipt I audited is identical to the one committed at `86dc15b4` (both hash to
`adfec88df9c422ec…`), so this audit is of the committed artifact, not of a working copy.

## My own verdict is bound into the freeze, by hash

The receipt records `external_pass_document_sha256: a4e9d738671099316faa9d8498233db72dfb3676fa585654946e644f656bdd7a`
and `external_pass_origin_commit: d2aa607d`. I recomputed the digest of my PASS document as it
exists in my own commit `d2aa607d`: it matches exactly. The freeze does not merely assert that
an external review happened — it names which review, in which commit, with which bytes.

## The status contradiction is resolved correctly

The frozen JSON still carries `status: DRAFT_PROSPECTIVE_UNOPENED_NOT_AUTHORITY` in its body.
That is deliberate and it is the right call: editing that field would have changed the bytes
and invalidated the external review of them. Authority lives in the separate receipt, and
`load_authority` enforces the binding through six chained conditions:

* the body must still carry the reviewed content marker;
* a separate freeze receipt must exist;
* `receipt.contract_sha256` must equal the hash of the contract bytes;
* `receipt.status` must be `FROZEN_PROSPECTIVE_UNOPENED`;
* `receipt.reviewed_contract_internal_status` must equal the body's marker;
* both `fresh_development_roots_opened` and `confirmation_roots_opened` must be `False`.

A tampered body, a swapped receipt or a receipt from a different contract all fail closed.

## Custody state at the freeze

| Item | State |
|---|---|
| Frozen contract | `bb92a2cb…`, byte-identical to the reviewed draft |
| Authoritative status | `FROZEN_PROSPECTIVE_UNOPENED`, in the separate receipt |
| `data_splits.opened` | `false` |
| Training / selection roots | closed |
| Confirmation roots 7670201-7670264 | closed, `confirmation_return_observed_before_freeze: false` |
| Development seeds 7672101-7672105 | closed |
| Instrument preflight | closed by a one-way door: `if FREEZE_RECEIPT_PATH.exists(): raise` |
| Comparator c256 | bound by its own receipt hash |

## Authorization

Run the `static-bar` step, which opens the selection roots only, then the development workers.
Confirmation stays sealed until a prospective power audit.

What I will audit on the development run, so it is stated in advance rather than invented
afterwards:

1. the development opening receipt is written **before** any root is materialized, and names
   the roots and seeds it opens;
2. every worker passes the full static-bar custody chain, and no worker builds its own bar;
3. no `REJECTED_OVER_CAP` unit appears in any eligible row set;
4. checkpoint selection touches only the selection split, never 7670201-7670264;
5. the four factorial arms of each checkpoint carry one identical checkpoint hash;
6. the service ledger is complete on every row;
7. the recovery policy is honoured if any worker dies — new directory, failed attempt
   preserved, exactly one complete attempt eligible.
