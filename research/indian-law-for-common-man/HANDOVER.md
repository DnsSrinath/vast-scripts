# Handover: Indian Law for the Common Man research

Date of handover: 5 September 2026
Branch: `claude/indian-epco-laws-research-a48ydn` (pushed, no pull request opened)
Folder: `research/indian-law-for-common-man/`

## 1. The original goal

Answer "how many laws does India have in total", read the main law books, and produce a deep but simple analysis that any citizen of India can understand. Research only, no coding. One architect model planned and cross-checked; Opus and Sonnet research agents did parallel web research.

## 2. What is finished and reviewed

| File | Status |
|---|---|
| `README.md` | Done. Index page with the verified headline counts and the chapter plan. |
| `02-constitution-and-rights.md` | Done and reviewed. Preamble, Fundamental Rights, Duties, Directive Principles, government structure, Seventh Schedule, emergencies, landmark cases, rights in ten everyday situations, sources. |

## 3. Headline findings (verified)

| Question | Answer as of Sept 2026 | Source quality |
|---|---|---|
| Constitution | 1. About 448 Articles, 25 Parts, 12 Schedules. 106 amendments enacted. | Verified |
| Central Acts in force | 859 (845 live plus 14 "spent"), excluding annual Finance Acts | Verified (Ministry of Law and Justice / India Code list, cited via Wikipedia, June 2026) |
| Central Acts repealed since 2014 | About 1,500. Latest: Repealing and Amending Act 2025 (assent 20 Dec 2025) repealed 71 Acts. | Verified (PRS, Drishti, SCC Online) |
| State Acts in force | Several thousand across 28 states and UTs. No official all-India total. Maharashtra alone has over 1,000 laws in force. | Estimate (Vidhi Centre) |
| Rules and regulations under Acts | Tens of thousands. No official total. | Estimate |
| Pending constitutional amendments | 130th Amendment Bill 2025 (removal of ministers detained 30 days) is in a Joint Parliamentary Committee. 131st Amendment Bill 2026 (seat expansion and delimitation to operationalise women's reservation) was DEFEATED in the Lok Sabha, 298 for, 230 against, 352 needed. | Verified (PRS, LiveLaw, Outlook) |
| Supreme Court strength | Raised from 34 to 38 judges in May 2026 (reported by the legal-machinery research agent; confirm against the Gazette before publishing). | Reported, not independently confirmed |

Honest one-line answer: one Constitution, roughly 860 central laws, a few thousand state laws, and a very large body of rules under them. Nobody publishes one exact grand total.

## 4. The "EPCO" question

There is no Indian law called EPCO. Most likely intended meanings:
- IPC, the Indian Penal Code 1860, replaced on 1 July 2024 by the Bharatiya Nyaya Sanhita 2023.
- EPFO, the Employees' Provident Fund Organisation and the rules under the EPF and Miscellaneous Provisions Act 1952.
- EPCO, the Environmental Planning and Coordination Organisation, a Madhya Pradesh government body in Bhopal.
A research agent was assigned to settle this and write chapter 6. Its output had not arrived when work stopped.

## 5. Unreviewed drafts saved in `drafts/`

These came from research agents and were copied here so they survive the ephemeral session. They have NOT been reviewed by the architect. Treat every fact as unverified until checked.

| File | State |
|---|---|
| `drafts/04-how-law-works.DRAFT-complete-unreviewed.md` | Agent reported complete: 325 lines, 9 sections plus sources. Covers types of law, how a law is made, enforcers, courts, case journeys, digital access, legal aid, Centre vs State, language of law. Agent flagged as soft: e-SCR translation counts, BNSS timelines, Pre-Legislative Consultation statistics, the "20 states with Right to Public Services Acts" figure, Lok Adalat statistics. |
| `drafts/03-law-books-explained.DRAFT-partial.md` | Agent still writing when stopped (418 lines at copy time). Covers criminal laws (BNS, BNSS, BSA) and civil laws up to the RTI Act. Family law, labour codes, tax and company law sections may be missing. |
| `drafts/00-architect-crosscheck-notes.md` | Architect's own verification notes with sources. |
| `drafts/01-count-of-laws.DRAFT-complete-unreviewed.md` | Arrived after the stop. Agent reported complete: 705 lines, every figure tagged verified-primary, verified-secondary or estimate, with a list of seven items to re-confirm. Key updates it brings: India Code's relaunched portal (13 Aug 2026) showed 836 central Acts, 1,410 state Acts and 77,072 sections on 28 Aug 2026, slightly below the 859 list figure; 6,779 central Acts enacted since 1834; the Jan Vishwas Act 2026 (assent 7 Apr 2026) decriminalised 717 provisions across 79 central Acts after the 2025 Bill was withdrawn; Law Commission has 289 reports; oldest laws in force are two 1836 Bengal Acts; the state-Act total is an unresolved contradiction between India Code's 1,410 and five states alone listing 3,453. |

## 6. Not started or not received

- Chapter 5, forty everyday situations with helplines (agent was running; no file received).
- Chapter 6, reforms, simplification, EPCO disambiguation, 10-point plan (agent was running; no file received).
- Chapter 0, "start here": the roughly 30 laws that touch every citizen. Architect synthesis, not begun.
- Chapter 7, consolidated sources and where to read laws free. Not begun.

## 7. How to resume

1. Check whether the remaining agents' files ever landed in the session scratchpad (`scratchpad/research/05-everyday-situations.md`, `06-reforms-and-epco.md`). If the container is gone, re-run those two research briefs; the prompts are recorded in the session transcript.
2. Review each draft against its own sources list, fix dates, then move it from `drafts/` to the chapter filename listed in `README.md`.
3. Write chapter 0 and chapter 7 from the finished chapters.
4. Commit and push to the same branch. Open a pull request only if asked.

## 8. Method notes and cautions

- The sandbox blocks direct fetches of indiacode.nic.in and wikipedia.org. All verification used search-result snippets that cite those sources, not the pages themselves. Re-check the 859 figure on India Code directly before publishing.
- Uttarakhand's Uniform Civil Code was passed in 2024 and came into force on 27 January 2025. One draft originally said 2024; fixed in chapter 2, check the others.
- Nothing in this folder is legal advice. Free legal aid helpline: 15100 (NALSA).
