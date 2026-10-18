"""Synthetic enterprise-RAG prompts.

A fixed policy document (~550 words) plus deterministic knowledge-base records
used to pad the system prompt to a target length. Everything is seeded, so the
same arguments always produce the same token sequence -- which is exactly what
prefix caching keys on.
"""

import functools
import random

from src import config

POLICY_DOC = """\
You are the internal support assistant for Halcyon Freight, a mid-sized logistics
company operating 14 distribution centers across North America. Answer employee and
customer-service questions using ONLY the reference material below. If the answer is
not in the material, say so and suggest the correct escalation channel. Be concise,
cite the section number you relied on, and never invent shipment data.

SECTION 1 - SHIPPING TIERS. Halcyon offers three service tiers. Standard Ground
delivers within five to seven business days and is the default for all parcels under
70 pounds. Expedited delivers within two to three business days and carries a 35
percent surcharge over Standard. Priority Overnight guarantees delivery by 10:30 AM
the next business day for addresses inside the primary coverage zone; outside that
zone the guarantee extends to end of day. Priority shipments must be tendered before
the 4:00 PM local cutoff at the origin facility.

SECTION 2 - DIMENSIONAL WEIGHT. Billable weight is the greater of actual weight and
dimensional weight. Dimensional weight is length times width times height in inches,
divided by 139 for domestic shipments and by 166 for cross-border shipments to Canada.
Any parcel whose longest side exceeds 96 inches is classified as oversize and is
billed a flat oversize handling fee in addition to its billable weight.

SECTION 3 - CLAIMS. Customers may file a damage claim within 30 calendar days of the
delivery date and a loss claim within 60 calendar days of the ship date. Claims require
the tracking number, photos of the outer packaging and contents, and the commercial
invoice establishing declared value. Standard liability is capped at 100 dollars per
package unless the shipper purchased declared-value coverage. Claims above 5,000
dollars are routed to the Risk and Compliance team and require a manager's sign-off.

SECTION 4 - HAZARDOUS MATERIALS. Halcyon accepts limited-quantity lithium batteries,
consumer aerosols, and ORM-D materials on Standard Ground only. Fully regulated
hazardous materials, including Class 1 explosives and Class 7 radioactive materials,
are never accepted. Shippers must present a valid safety data sheet on request.
Misdeclared hazardous shipments are held at the origin facility and the account is
flagged for review.

SECTION 5 - RETURNS AND REDELIVERY. Three delivery attempts are made for packages
requiring a signature. After the third failed attempt the package is held for five
business days at the nearest facility, then returned to the shipper at the shipper's
expense. Recipients may request a hold-for-pickup or a one-time address change through
the self-service portal for a fixed redirect fee, provided the package has not yet
been loaded onto the final delivery vehicle.

SECTION 6 - ESCALATIONS. Billing disputes go to accounts receivable through the
finance ticket queue. Service failures on Priority Overnight are eligible for an
automatic money-back guarantee if the customer requests it within 15 days. Suspected
fraud, theft, or tampering must be reported to Corporate Security immediately, and
customer-facing staff must not discuss the details of an open security investigation.

SECTION 7 - KNOWLEDGE BASE RECORDS. The records below describe individual facilities,
lanes, and customer accounts. Use them to answer specific operational questions.
"""

_CITIES = ["Atlanta", "Dallas", "Denver", "Chicago", "Phoenix", "Seattle", "Toronto",
           "Memphis", "Columbus", "Reno", "Charlotte", "Kansas City", "Salt Lake City",
           "Louisville", "Montreal", "Sacramento"]
_INDUSTRIES = ["medical devices", "consumer electronics", "auto parts", "apparel",
               "specialty foods", "industrial tools", "cosmetics", "furniture"]

QUESTIONS = [
    "What is the dimensional weight divisor for a shipment to Canada?",
    "A customer's package was damaged and delivered 40 days ago. Can they still file a claim?",
    "Can we ship a fully regulated Class 7 material on Priority Overnight?",
    "What happens after the third failed delivery attempt on a signature package?",
    "Who must approve a claim worth 7,200 dollars?",
    "What is the surcharge for Expedited compared to Standard Ground?",
    "Which facility handles the most daily volume according to the records?",
    "How long does a customer have to request the Priority Overnight money-back guarantee?",
    "Is a 100-inch parcel considered oversize?",
    "What is the tender cutoff time for Priority Overnight shipments?",
]


def _record(i: int, rng: random.Random) -> str:
    city, other = rng.sample(_CITIES, 2)
    return (
        f"Record {i:04d}: Facility HF-{100 + i} in {city} processes roughly "
        f"{rng.randint(2, 40) * 500} parcels per day, operates {rng.randint(2, 6)} dock shifts, "
        f"and serves a key account in {rng.choice(_INDUSTRIES)} with an on-time rate of "
        f"{rng.randint(880, 999) / 10} percent. Its primary outbound lane runs to {other} "
        f"with a transit time of {rng.randint(1, 5)} days."
    )


@functools.lru_cache(maxsize=1)
def _tokenizer():
    try:
        from transformers import AutoTokenizer  # installed with vllm
        return AutoTokenizer.from_pretrained(config.MODEL)
    except Exception:
        return None


def count_tokens(text: str) -> int:
    tok = _tokenizer()
    if tok is None:  # rough fallback; Qwen splits digits into single tokens
        return int(len(text.split()) * 1.6)
    return len(tok.encode(text, add_special_tokens=False))


def system_prompt(target_tokens: int = 800, nonce: str = "") -> str:
    """Policy doc + KB records, padded to ~target_tokens with the model's own tokenizer.

    `nonce` is prepended to the very first line. Changing it changes the hash of
    the first KV block, so *nothing* downstream can be served from cache -- this is
    how we guarantee a true cold start without restarting the server.
    """
    rng = random.Random(0)
    parts = [f"[session {nonce}]\n" if nonce else "", POLICY_DOC]
    n = count_tokens("".join(parts))
    i = 0
    while n < target_tokens:
        rec = _record(i, rng) + "\n"
        parts.append(rec)
        n += count_tokens(rec)
        i += 1
    return "".join(parts)


def chat(system: str, question: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]
