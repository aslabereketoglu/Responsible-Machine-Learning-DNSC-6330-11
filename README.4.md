
Part A: Distribution drift Compute PSI and KS for numeric features Compute MMD in encoded feature space Compare train vs. test score distributions

The distribution drift analysis shows minimal differences between the training and test datasets across numeric features, encoded feature space, and predicted scores.

For the raw numeric features (priors_count and two_year_recid), the PSI values are extremely low (0.0104 and 0.0008), well below the 0.10 threshold. This indicates that the feature distributions remain stable between training and test sets. Similarly, the KS statistics are small and the p-values are high (greater than 0.05), meaning we fail to reject the null hypothesis that the distributions are the same. Together, these results suggest no statistically significant input drift.

In the high-dimensional encoded feature space, the MMD² value is approximately zero (-0.000272), indicating that the joint distribution of features after preprocessing is nearly identical across train and test sets. This further confirms the absence of meaningful distributional shift at a global level.

The score distribution comparison across both models (Logistic Regression and Gradient-Boosted Tree) also shows negligible drift. The mean predicted probabilities are nearly identical between train and test sets, and both PSI and KS statistics for scores are extremely small with high p-values. This suggests that model outputs are stable and consistent across datasets, indicating no evidence of prediction drift.

These results suggest that the model is being evaluated under conditions where the training and test data are drawn from very similar distributions. While this is ideal for measuring baseline performance, it also means that this evaluation may not reflect real-world deployment conditions, where distribution shift is common.

Although PSI, KS, and MMD indicate stability, they only detect observable distribution differences. They do not capture:

Concept drift (changes in the relationship between features and the target) Spurious correlations that may still exist even if distributions match Subgroup-specific drift, where certain populations may experience different distributions

Since no significant drift is detected, no immediate retraining is required. However, this stability should be interpreted cautiously. The absence of drift suggests that the test set is very similar to the training data, which may lead to overly optimistic estimates of model reliability.

In a real deployment setting, continuous monitoring using PSI, KS, and performance metrics would still be necessary to detect future drift. Additionally, further analysis (e.g., robustness testing and slice-based evaluation) is required to ensure the model remains reliable across subpopulations and under changing conditions.

Part B: Generalization Compare train vs. test AUC, accuracy, and log loss Diagnose overfitting using performance gaps 
The generalization analysis evaluates how well the models perform on unseen data by comparing training and test metrics, including accuracy, AUC, and log loss.

For Logistic Regression, the differences between training and test performance are negligible. The accuracy gap is very small (0.0017), and the AUC gap is slightly negative (-0.0037), indicating that test performance is essentially the same as training performance. The log loss gap (0.0082) is also minimal. These results suggest that the model generalizes well, with no evidence of overfitting.

For the Gradient-Boosted Tree, the gaps are more noticeable. The accuracy gap (0.0250) and AUC gap (0.0245) indicate that the model performs better on the training data than on the test data. Additionally, the log loss gap (-0.0380) suggests worse calibration on unseen data. These differences indicate mild overfitting, where the model has learned patterns specific to the training data that do not fully generalize.

The generalization gap measures the difference between training (empirical risk) and test performance (true risk). A larger gap indicates that the model’s performance on training data cannot be trusted in deployment.

In this case:

Logistic Regression shows a near-zero generalization gap → low variance, stable generalization Gradient-Boosted Tree shows a positive gap → higher variance, mild overfitting

This aligns with the bias–variance tradeoff discussed in class: simpler models tend to generalize better, while more complex models are more prone to overfitting.

While performance gaps help detect overfitting, they do not capture:

Whether the model is relying on spurious correlations Whether performance varies across subgroups Whether the model will remain stable under distribution shift

Thus, good generalization on a test set does not guarantee robustness in real-world deployment.

Logistic Regression appears reliable for deployment from a generalization perspective due to its stability across train and test sets. Gradient-Boosted Tree shows mild overfitting and should be monitored, especially if deployed in changing environments. Continued monitoring of performance metrics over time is necessary to detect future degradation.

Part C: Spurious-correlation probe Run counterfactual swaps on selected attributes Measure change in predicted probabilities 
The counterfactual swap analysis evaluates whether the model relies on potentially spurious or unstable relationships by measuring how predicted probabilities change when selected attributes are altered while holding all other features constant.

For Logistic Regression, swapping race from African-American to Caucasian results in an average probability shift of 0.0806, while gender swaps produce a smaller shift of 0.0251. Crime type changes also lead to moderate shifts (0.0410). For the Gradient-Boosted Tree, the effects are stronger: race swaps produce a shift of 0.0976, and gender swaps lead to a substantial shift of 0.0888. These results indicate that predictions are meaningfully sensitive to changes in demographic attributes.

Spurious correlations occur when a model learns associations that are present in the training data but do not reflect stable or causal relationships. Counterfactual testing is a diagnostic tool to detect such behavior by checking whether predictions change when non-causal or sensitive attributes are modified.

The observed probability shifts suggest that both models, especially the Gradient-Boosted Tree, may be relying on shortcut features or proxy relationships involving race and gender. This is consistent with the concept of shortcut learning, where a model achieves good performance by exploiting correlations that may not generalize or may encode bias.

The relatively large probability shifts, particularly for race and gender, indicate that these attributes (or correlated proxies) have a non-trivial influence on predictions. This raises concerns that the model may not be relying solely on stable, task-relevant features, but instead on patterns that could reflect underlying biases in the data.

The stronger sensitivity observed in the Gradient-Boosted Tree compared to Logistic Regression suggests that higher-capacity models are more prone to capturing such spurious relationships.

While counterfactual swaps reveal sensitivity to specific attributes, they do not determine:

Whether the observed relationships are causally justified Whether these shifts result in unfair outcomes across subgroups Whether these patterns persist under distribution shift or over time

The results suggest a potential risk of spurious correlation and proxy discrimination, particularly for race and gender. This warrants further investigation before deployment.

Recommended actions include: Conduct slice-based evaluation to assess subgroup performance disparities Use SHAP or feature attribution methods to better understand feature reliance Evaluate whether sensitive attributes or their proxies should be restricted or controlled Monitor prediction stability under attribute perturbations as part of a robustness audit

Part D: Robustness Stress test priors count Produce ICE curves and sensitivity summaries
The robustness analysis evaluates how model predictions respond to controlled perturbations of a key feature (priors_count), using stress testing, ICE curves, and sensitivity metrics.

Stress Test Results

As priors_count increases, both models show a consistent rise in predicted risk. For Logistic Regression, the mean predicted probability increases from 0.4476 to 0.8435 as the feature is incremented, and the share of high-risk predictions rises substantially (from 42.3% to 92.5%). The Gradient-Boosted Tree exhibits a similar pattern, though slightly less extreme at higher perturbations.

This indicates that both models are highly sensitive to changes in priors_count, and that this feature has a strong influence on predictions. Importantly, the changes are monotonic and smooth for Logistic Regression, suggesting stable behavior under stress.

The ICE curves reveal how predictions change for individual observations:

For Logistic Regression, the curves are smooth and consistently increasing, indicating that the model responds predictably and uniformly across individuals. This suggests stable and interpretable behavior, where the effect of priors_count is consistent. For the Gradient-Boosted Tree, the ICE curves are irregular and non-smooth, with abrupt jumps and variability across individuals. This indicates heterogeneous responses, meaning the same change in priors_count can lead to very different prediction shifts depending on the individual.

According to the lecture, such variability can signal instability and potential shortcut learning, where the model reacts unpredictably under small input changes.

The sensitivity index values (~0.0488 for Logistic Regression and ~0.0412 for Gradient-Boosted Tree) indicate that priors_count explains a meaningful portion of prediction variance.

Additionally, the wide prediction ranges (≈0.70+) confirm that this feature can drive large changes in model outputs. While this may be expected if the feature is genuinely predictive, the lecture notes warn that high sensitivity without strong domain justification may indicate over-reliance on a single feature.

The lecture defines robustness as the model’s ability to degrade gracefully under perturbations rather than failing unpredictably.

Logistic Regression demonstrates graceful degradation: predictions change smoothly and consistently Gradient-Boosted Tree shows less stable behavior, with irregular responses across individuals

This analysis does not determine:

Whether the sensitivity to priors_count is causally justified Whether this sensitivity leads to unfair outcomes across subgroups How the model behaves under other types of perturbations or real-world shifts

Logistic Regression appears more robust due to its stable and predictable response to feature changes Gradient-Boosted Tree shows signs of instability and should be used with caution The strong influence of priors_count suggests a need to validate its role with domain knowledge Sensitivity analysis results should be documented as part of a model risk management (MRM) report, as recommended in the lecture

Part E: Slice-based evaluation Compare performance by race, gender, and age slices

The slice-based evaluation examines model performance across demographic subgroups (race, gender, and age) to identify disparities that may be hidden in aggregate metrics.

Performance Across Age Groups

Both models show variation across age slices. For individuals under 25, performance is noticeably worse (e.g., Logistic Regression AUC = 0.7175, GBT AUC = 0.7313) compared to the 25–45 group (AUC ≈ 0.83). Additionally, younger individuals have significantly higher predicted risk scores and higher positive prediction rates.

This suggests that the models may systematically assign higher risk to younger individuals, indicating a potential age-related bias or structural pattern in the data.

Performance differences between males and females are moderate. For Logistic Regression, females have slightly lower accuracy and higher Brier scores, indicating worse calibration. Similar patterns appear in the Gradient-Boosted Tree.

Although disparities are not extreme, the consistent performance gap suggests unequal model behavior across gender groups, which may warrant further monitoring.

The largest disparities appear across race groups:

African-American individuals have higher predicted risk scores and positive prediction rates compared to other groups. Caucasian individuals have significantly lower predicted risk and lower positive prediction rates. Some groups (e.g., Asian, Native American) have extremely small sample sizes, leading to unstable and extreme metric values (e.g., AUC = 1.000 or FPR = 1.000).

These findings indicate substantial differences in model behavior across racial groups, particularly in prediction rates and error metrics (FPR and FNR). According to the lecture, such disparities are critical because aggregate performance can mask subgroup harms.

The lecture emphasizes that slice-based evaluation is essential for identifying hidden failure modes. A model may perform well overall but fail for specific subpopulations.

In this case:

Aggregate performance appears strong (Part B) But slice-level results reveal heterogeneous performance and disparities

This aligns with the principle that “failure on a slice is a failure mode”, even if overall metrics are high.

What this analysis shows Performance is not uniform across subgroups Certain groups (e.g., African-American, younger individuals) experience higher predicted risk and different error rates Small subgroups produce unstable metrics, highlighting data limitations

What this analysis misses Whether disparities are due to true underlying risk differences or bias Whether observed differences violate fairness criteria (e.g., equalized odds) How these disparities evolve under distribution shift or over time

The observed subgroup disparities indicate potential fairness and reliability risks Models should not be evaluated solely on aggregate performance Additional fairness analysis (e.g., AIR, FPR/FNR parity) is recommended Small sample groups should be treated cautiously, as their metrics are unstable

Recommended actions:

Conduct formal fairness audits using disparity metrics Consider rebalancing or augmenting underrepresented groups Monitor subgroup performance continuously in deployment Document these disparities as part of a model risk governance process
