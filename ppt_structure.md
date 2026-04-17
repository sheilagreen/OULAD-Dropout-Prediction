# Presentation Outline — Predicting Student Dropout in Online Courses

**Format:** 10 min presentation + 5 min live demo = 15 min total | **Team:** 3 presenters | **Q&A:** 5 min after

Each presenter gets ~3:20 of speaking time. The story arc: **Problem → Data & Approach → Results & Impact**.

---

## 🎬 Slide 1 — Title Slide *(15 sec, Presenter 1 opens)*

- **Title:** Predicting Student Dropout in Online Courses: An Early-Warning System
- Team names, DATA 6545, Spring 2026
- One-line hook: *"Catching at-risk students by Day 30 — while there's still time to help."*

---

## 🎯 PART 1 — THE PROBLEM & THE DATA *(Presenter 1 — ~3 min, slides 2–5)*

### Slide 2 — Why This Matters *(45 sec)*

- Online courses have chronic high withdrawal rates (~31% in our dataset)
- Every dropout = lost revenue + lost learning outcome + wasted outreach dollars
- **The core idea:** if we can flag at-risk students *early*, interventions (tutoring, check-ins, reminders) can actually work
- Stakeholders: platform admins & student support teams

### Slide 3 — The Business Question *(30 sec)*

- One framed question: *"Using only the first 30 days of course data, can we predict who will withdraw — early enough to intervene?"*
- Why Day 30? Sweet spot between enough behavioral signal and enough runway to act
- Primary metric: **Recall** (missing an at-risk student is the costly error)

### Slide 4 — The Data: OULAD *(45 sec)*

- Open University Learning Analytics Dataset
- **32,593 student-course enrollments** across 4 tables
- Demographics + registration + VLE clickstream (10.6M records) + assessments (174K records)
- Filtered to only what's visible by Day 30 → 2.98M click events, 26.5K submissions

### Slide 5 — What We Engineered *(60 sec)*

- 36 features after encoding, grouped into 3 families:
  - **Enrollment & demographics** (credits, registration timing, prior attempts, education, region, IMD band)
  - **Early VLE engagement** — `total_clicks`, `active_days`, `avg_clicks_day`, `last_activity`
  - **Early assessment behavior** — `avg_score`, `num_submitted`, `num_unsubmitted`
- Strongest EDA signal: `last_activity` (r = −0.41), `avg_score` (r = −0.34) — both clearly separate stayers from withdrawers
- *Visual:* box plots of total clicks / active days / submissions by outcome

---

## 🛠 PART 2 — MODELING & EVALUATION *(Presenter 2 — ~3:30, slides 6–9)*

### Slide 6 — Modeling Approach *(45 sec)*

- 80/20 stratified split, class imbalance handled via class weighting + threshold tuning
- 6 tracked experiments in MLflow: Logistic Regression → Random Forest (default + balanced) → Gradient Boosting → tuned variants
- RandomizedSearchCV on top candidates, scoring on recall

### Slide 7 — Model Comparison *(60 sec)*

- Show the 6-row results table (Accuracy / Recall / Precision / F1 / AUC)
- **Winner: Tuned Balanced Random Forest** — Recall 0.61, Precision 0.65, **F1 0.63, AUC 0.80**
- Key insight: the baseline LR has higher raw recall (0.63) but terrible precision — too many false alarms. The tuned RF is the *operationally useful* model.

### Slide 8 — The Threshold Decision *(75 sec) — this is the story slide*

- Default 0.50 threshold misses too many at-risk students
- Threshold sensitivity table (0.20 / 0.30 / 0.40 / 0.50):

| Threshold | Recall | Precision | Students Flagged |
|-----------|--------|-----------|------------------|
| 0.20      | 81%    | 47%       | 3,484            |
| **0.30**  | **67%**| **59%**   | **2,306**        |
| 0.40      | 57%    | 69%       | 1,680            |
| 0.50      | 49%    | 76%       | 1,306            |

- **We chose 0.30** → catch 2 in 3 withdrawals while keeping outreach volume manageable
- Takeaway: model tuning is a *business decision*, not just a math decision

### Slide 9 — Why the Model Makes Sense (SHAP) *(30 sec)*

- Top predictors: `avg_score`, `last_activity`, `studied_credits`, `num_submitted`, `avg_clicks_day`
- Directionality matches intuition: low scores + stale activity + overcommitment → higher risk
- Makes the model *defensible* to advisors who need to understand why a student was flagged

---

## 📊 PART 3 — IMPACT, LIMITATIONS & DEMO *(Presenter 3 — ~3:15, slides 10–13, then hands to demo)*

### Slide 10 — From Metrics to Real-World Impact *(60 sec)*

- Intervention simulation at threshold 0.30:
  - 2,306 students flagged on our test set
  - 1,352 would have actually withdrawn
  - At a conservative 20% intervention success rate → **~270 students retained from one test sample alone**
- Reframe: this isn't "an F1 of 0.63" — it's *hundreds of students kept on track*

### Slide 11 — Honest Limitations *(45 sec)*

- **Single-institution data** — won't generalize without retraining
- **Static features** — we aggregate, we don't capture *trajectories* (e.g., declining engagement)
- **ID-level merge** — small leakage risk for students in multiple courses
- **Rolling window finding:** some students become at-risk *after* Day 30 → a single-window model misses them. A biweekly re-scoring cadence would fix this.

### Slide 12 — Ethical Considerations *(30 sec)*

- Risk of labeling / stigmatizing flagged students
- Model should be *decision support*, not an automated action trigger
- Demographic features (IMD band, region) require fairness monitoring in production
- Students should know they can be flagged and have recourse

### Slide 13 — What We Built *(30 sec)*

- ✅ Trained & tuned model with versioned artifacts
- ✅ MLflow experiment tracking (6 runs)
- ✅ Flask API prototype for real-time scoring
- ✅ Full documentation of features, thresholds, and trade-offs
- **Next up: 5-minute live demo**

### Slide 14 — Demo Transition *(handoff slide, ~10 sec)*

- "Let's see it work on a student record." → demo begins

---

## 🎯 Story Arc Summary

| Section              | Presenter | Emotional Beat                                         |
|----------------------|-----------|--------------------------------------------------------|
| Problem + Data       | P1        | *"Here's a real problem with real stakes"*             |
| Modeling + Threshold | P2        | *"Here's the rigorous, defensible choice we made"*     |
| Impact + Demo        | P3        | *"Here's what it actually does for 270 students"*      |

## ⏱ Timing Budget

- Total speaking: ~10 min | Demo: 5 min | Q&A: 5 min (after)
- Built-in buffer: ~30 sec for transitions between presenters
