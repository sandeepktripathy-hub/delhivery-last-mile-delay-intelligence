# Delhivery Last-Mile Delay Intelligence Pipeline

IIT Roorkee E&ICT Academy — Artificial Intelligence and Machine Learning Capstone Project

This project builds an end-to-end delay intelligence pipeline for synthetic Delhivery shipment data. It addresses three operational questions:

1. How long will a parcel take to be delivered?
2. Which parcels are likely to miss their promised delivery time?
3. Which hubs require different operational playbooks based on their actual performance?

The pipeline covers data cleaning and feature engineering, delivery-hours regression, delay classification and alert-threshold selection, hub segmentation, model packaging, and scoring of new shipments into operational actions.

## How to run

From a fresh clone of the repository, run the following commands in order:

```bash
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`generate_data.py` regenerates the three synthetic source datasets under `data/`.

`part_a_clean_eda.py` creates the cleaned modelling table and Part A outputs.

`part_b_regression.py` generates the regression experiments and evaluation outputs.

`part_c_classification.py` generates the delay-classification, feature-selection, tuning, threshold and final-test outputs.

`part_d_segments.py` creates the hub profiles, K-Means segmentation outputs and tier-versus-segment comparison.

`package_model.py` regenerates `models/delay_model.pkl`, along with the stored threshold, feature order and imputation values.

`predict_delay.py` scores the 48 new shipments and writes the final operational action list.

The committed `models/delay_model.pkl` should be loaded using the scikit-learn version pinned in `requirements.txt`. If the installed scikit-learn version differs, run `python package_model.py` to regenerate the pickle before scoring.

## Data findings

After cleaning, the modelling table contains 1,200 unique historical shipments with no missing values across the 17 model features, `delivery_hours`, and `is_delayed`.

### Delay by service type

Express shipments had the highest delay rate at **0.4897**, with 119 delayed shipments out of 243. Standard shipments had a delay rate of **0.2056**, while Economy shipments had the lowest rate at **0.0857**.

The lower Economy delay rate should be interpreted in the context of its longer 96-hour promise, while Express has a much tighter 36-hour promise.

### Delay by hub tier

Delay risk increased strongly with hub tier:

- Tier 1: **0.0400**
- Tier 2: **0.2150**
- Tier 3: **0.4400**

This shows that higher-tier-number hubs in this synthetic network are materially more exposed to delayed deliveries.

### Delay by distance band

Delay rate also rose sharply with shipment distance:

- 0–200 km: **0.0219**
- 201–400 km: **0.1368**
- 401–600 km: **0.4364**
- 601+ km: **0.8523**

The 601+ km group therefore has the highest observed delay risk.

### Strongest correlations with delivery hours

The three strongest correlations with `delivery_hours`, excluding the target itself, were:

- `promised_hours`: **0.8372**
- `distance_km`: **0.4013**
- `n_stops`: **0.3247**

These results indicate that service promise, route distance and route complexity are the strongest linear drivers of delivery duration in this dataset.

### Declared-value transformation

The filled `declared_value_inr` column has a mean of **1533.0542** and a median of **1121.5000**, showing right-skew.

After applying the natural logarithm, the mean becomes **7.0277** and the median **7.0224**, making the distribution substantially more balanced. This is why `log_declared_value` is used as the modelling feature.

## Model results — regression

The mean baseline produced a validation R² of **-0.0212**. A negative R² means that simply predicting the training-set mean performs slightly worse than using the validation-set mean as a reference, so the baseline has no useful predictive power.

Among the candidate models, Linear Regression achieved the lowest validation RMSE at **4.8912 hours**, narrowly ahead of Gradient Boosting at **4.9165 hours**. Linear Regression was therefore selected as the final model by the validation-RMSE rule.

The final Linear Regression model achieved a test RMSE of **4.5235 hours** and a test R² of **0.9493**.

### Decision-tree depth diagnostic

At depth 1, the training RMSE was **11.6334** and validation RMSE was **12.1324**. Both are high and close together, indicating high bias and underfitting.

Validation performance improved as the tree became deeper, reaching its best validation RMSE of **6.2796** at depth 6.

With no depth limit, training RMSE fell to **0.0000**, while validation RMSE rose to **7.8368**. This large gap is a clear sign of high variance and overfitting.

### Gradient-boosting learning rate

The learning-rate experiment produced:

- 0.01: validation R² **0.7127**
- 0.10: validation R² **0.9306**
- 1.00: validation R² **0.8440**

A learning rate of 0.01 learns too slowly with only 100 estimators. A learning rate of 1.0 takes updates that are too aggressive and loses validation performance. The intermediate value of 0.10 provides the best balance.

### Cross-validation

For the two best validation models:

- Linear Regression: mean R² **0.9261**, standard deviation **0.0048**
- Gradient Boosting: mean R² **0.9353**, standard deviation **0.0045**

Cross-validation slightly favoured Gradient Boosting, while the single validation split slightly favoured Linear Regression.

For this assignment, Linear Regression is shipped because the stated final-model rule selects the candidate with the lowest validation RMSE. The cross-validation result is still important evidence that Gradient Boosting is a strong alternative.

## Model results — classification

The blind baseline predicts every shipment as on time. It achieved validation accuracy of **0.7667**, but recall and F1 were both **0.0000**, because it failed to identify any of the 42 delayed validation parcels. Accuracy alone is therefore misleading for this imbalanced problem.

The initial trained models achieved validation F1 scores of:

- Logistic Regression: **0.8046**
- Decision Tree, depth 6: **0.7470**
- Default Random Forest: **0.7848**

### Feature signal and noise

The seven highest F-score features were:

1. `distance_km`
2. `hub_tier`
3. `hub_daily_capacity`
4. `n_stops`
5. `promised_hours`
6. `rain_flag`
7. `sale_event_flag`

All five bookkeeping/noise columns ranked outside the top seven by F-score.

The five noise-column Random Forest importances were:

- `label_print_batch`: **0.0460**
- `scanner_build_no`: **0.0374**
- `tape_roll_no`: **0.0476**
- `sticker_serial_mod`: **0.0499**
- `shelf_slot_no`: **0.0436**

Although these bookkeeping fields receive some tree-based importance collectively, their weak F-scores support excluding them from the final seven-feature classifier.

### Select-K experiment

Validation F1 by number of selected features was:

- k=5: **0.6988**
- k=7: **0.8333**
- k=10: **0.7857**
- k=12: **0.8049**
- k=17: **0.7848**

The strongest validation result occurred at k=7, supporting the assignment's final use of seven selected features.

### Tuning result

Validation F1 was:

- Grid Search: **0.7750**
- Randomized Search: **0.7848**
- Default Forest on the same seven features: **0.8333**

The untuned seven-feature forest performed best on the 180-row validation set. This shows that a larger hyperparameter search does not automatically improve generalisation, especially when model choices are being compared on a relatively small validation sample.

### Alert threshold

The selected alert threshold is **0.5000**, because it is the highest tested threshold whose validation recall remains at least 0.80.

At threshold 0.5, validation precision is **0.7955** and recall is **0.8333**. Raising the threshold would reduce unnecessary alerts but would also cause recall to fall below the required operations level.

On the final test set, the model achieved:

- Precision: **0.6889**
- Recall: **0.7561**
- F1: **0.7209**
- Accuracy: **0.8667**

The test recall is lower than the validation recall, which is an important limitation to acknowledge. The test set contains only 41 delayed parcels, so each missed delayed shipment has a noticeable effect on the measured recall.

## Operations recommendation

The K-Means hub analysis produces three operational segments:

- Standard: 7 hubs, delay rate **0.0257**
- Watch: 8 hubs, delay rate **0.1550**
- Escalate: 9 hubs, delay rate **0.4600**

The Escalate hubs and their historical delay rates are:

- HUB13 Nagpur: **0.4600**
- HUB15 Kochi: **0.3600**
- HUB17 Patna: **0.4600**
- HUB18 Ranchi: **0.3800**
- HUB19 Guwahati: **0.6200**
- HUB20 Raipur: **0.4800**
- HUB21 Bhubaneswar: **0.4000**
- HUB23 Siliguri: **0.4400**
- HUB24 Agartala: **0.5400**

These hubs should receive the strongest operational playbook because their observed behaviour places them in the highest-delay operating segment.

### Tier-versus-segment mismatches

Four hubs behave differently from what their existing tier label suggests:

- HUB07 Kolkata: Tier 1 but Watch
- HUB13 Nagpur: Tier 2 but Escalate
- HUB15 Kochi: Tier 2 but Escalate
- HUB22 Dehradun: Tier 3 but Watch

The tier label should therefore be treated as a structural classification rather than a complete description of current operational performance. The behaviour-based segment provides an additional operational signal.

### Alert rule

The operational alert threshold is **0.5000**.

On the final test set, precision is **0.6889**, meaning most flagged shipments are genuinely delayed, while recall is **0.7561**, meaning the model catches a substantial majority of delayed parcels.

Among the 48 new shipments:

- **8** were flagged
- **6** require escalation to a hub manager
- **2** require a proactive customer call
- **40** require no action

### Why K = 3 is retained

The across-to-within separation ratios were:

- K=2: **2.8940**
- K=3: **5.3928**
- K=4: **6.3047**
- K=5: **6.8866**

Although the numerical separation continues to rise for K=4 and K=5, those solutions create very small clusters. K=4 has sizes **8-7-5-4**, while K=5 has **8-7-4-3-2**.

The business requires three operational playbooks, and K=3 produces well-separated groups of **9-8-7** hubs without creating clusters that are too small to justify their own operating model. K=3 is therefore retained.

## Production plan

### Production-readiness characteristics

**Accuracy:** Regression and classification performance are evaluated on held-out test data. The final regression model achieves test R² **0.9493**, while the final classifier achieves test F1 **0.7209**.

**Reliability:** `package_model.py` reloads the saved classifier and compares probabilities from the reloaded and in-memory models. The maximum absolute probability difference is **0.0000**.

**Reproducibility:** All random splits, forests and K-Means models use the fixed random states specified in the brief. `requirements.txt` pins the exact package versions used.

**Maintainability:** Shared cleaning and feature engineering are implemented once in `src/shipment_prep.py` and reused by training and prediction scripts.

**Consistency:** `predict_delay.py` uses the same `clean_shipments` and `build_features` functions used to prepare the historical data, and it uses the stored feature order and imputation medians.

**Monitorability:** The production process should log predictions, flags, actual outcomes once available, and hub-level delay behaviour so performance drift can be identified.

### Monitoring triggers

The following metrics should be monitored after deployment:

- Weekly delayed share of completed shipments: investigate if it changes by more than **0.05** from the recent four-week baseline.
- Weekly share of shipments flagged: investigate if it moves above **0.30** or below **0.10** for two consecutive weeks.
- Recall on shipments whose true outcomes are now known: retraining review if recall falls below **0.65**.
- Precision on matured flagged shipments: review the threshold if precision falls below **0.55**.
- Hub-level delay rate: operational review if a hub's rolling delay rate changes by more than **0.10** from its recent baseline.

### Retraining and drift

The model should be reviewed monthly and retrained at least quarterly, or earlier when monitoring triggers indicate material performance or data drift.

The 15–28 September sale period represented by `sale_event_flag` should be compared with similar promotional periods rather than normal weeks because sale events create a structurally different workload.

Data drift could appear as meaningful changes in shipment distance, route stops, service mix, COD share, parcel weight, hub capacity, rain exposure, sale-event intensity or the overall delayed share. Changes in these distributions can make historical model relationships less representative of current operations.

## Limitations

1. The entire dataset is synthetic. It is useful for demonstrating an AIML pipeline but cannot reproduce every operational complexity present in real logistics data.

2. The final test set contains only 41 delayed shipments. Classification precision, recall and F1 can therefore move materially when only a few parcels change classification.

3. The alert threshold was selected using one validation split. A production deployment should validate threshold stability over multiple time periods before fixing a long-term operational policy.

4. The model uses the variables available in the synthetic dataset and does not include other potentially important real-world drivers such as traffic conditions, live weather severity, route disruptions, courier availability or real-time hub congestion.

5. Hub segmentation is based on historical aggregate behaviour. Segment membership should be periodically recomputed because hub performance can improve or deteriorate over time.
