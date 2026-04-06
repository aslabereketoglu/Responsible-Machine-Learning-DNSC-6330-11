
Question 1: Compute AIR, ME, and SMD for race and sex separately using the solas-ai Python library. Confirm both produce identical results.

The AIR, ME, and SMD values computed using the solas-ai library match the manually computed results. Minor differences arise only from formatting conventions: solas reports percent difference and SMD in percentage form, while the manual implementation expresses them in decimal form. After adjusting for these differences, the results are identical

Code explanation: 

groups = ['African-American', 'Asian', 'Hispanic', 'Native American', 'Other']
reference = 'Caucasian'

def race_metrics(df, group_name):
    group_data = pd.DataFrame({
        group_name: (df['race'] == group_name),
        reference: (df['race'] == reference)
    })

    # AIR + ME
    air_obj = sd.adverse_impact_ratio(
        group_data=group_data,
        protected_groups=[group_name],
        reference_groups=[reference],
        group_categories=['race'],
        outcome=df['high_risk'],
        air_threshold=0.80,
        percent_difference_threshold=0
    )

    air_val = air_obj.summary_table.loc[group_name, 'AIR']
    me_val = -air_obj.summary_table.loc[group_name, 'Percent Difference Favorable']

    # SMD
    smd_obj = sd.standardized_mean_difference(
        group_data=group_data,
        protected_groups=[group_name],
        reference_groups=[reference],
        group_categories=['race'],
        outcome=df['decile_score'],
        smd_threshold=0.20
    )

    smd_val = smd_obj.summary_table.loc[group_name, 'SMD'] / 100

    return {
        'race': group_name,
        'AIR': air_val,
        'ME': me_val,
        'SMD': smd_val
    }

race_results = pd.DataFrame([race_metrics(df, g) for g in groups])

print("Race Results")
print(race_results.to_string(index=False))

The key idea of this code is that it computes fairness metrics (AIR, ME, and SMD) for each race by comparing that group to a reference group (Caucasian) using the Solas AI library. First, it creates a temporary binary dataset that identifies whether each observation belongs to the target group or the reference group, which is required input for Solas. Then, it calls adverse_impact_ratio() to calculate AIR and uses its output to derive ME (since Solas reports percent difference instead). Next, it calls standardized_mean_difference() to measure differences in continuous scores (decile_score), converting the result into decimal form. This process is repeated for each race group, and all results are collected into a final table. Overall, the code follows the fairness measurement framework from lecture compute per-group metrics relative to a reference group and report effect sizes like AIR, ME, and SMD to assess potential disparate impact.


Question 2: Build an intersectional analysis (race × sex). Report the worst-group AIR and interpret it. 

The intersectional analysis shows that the worst-performing subgroup is Other / Female, with an AIR of 0.235 relative to the reference group (Caucasian / Male). This value is far below the 0.80 threshold, indicating a substantial disparity in model outcomes for this subgroup. In practical terms, individuals in this group are flagged as high risk at only about 23.5% of the rate of the reference group. Because the subgroup size is relatively small (n = 58), this finding should be interpreted with some caution, but it still suggests meaningful intersectional disparity that would be masked by looking at race or sex separately


Code Explanation:

# Intersectional analysis -- race x sex
df['subgroup'] = df['race'] + ' / ' + df['sex']
# Keep subgroups with n >= 30
counts = df['subgroup'].value_counts()
valid_sg = counts[counts >= 30].index
df_sub = df[df['subgroup'].isin(valid_sg)].copy()
sub_rates = (df_sub.groupby('subgroup')['high_risk'].agg(['mean','count']).rename(columns={'mean':'selection_rate','count':'n'}).reset_index())
ref_rate = sub_rates.loc[sub_rates['subgroup']=='Caucasian / Male','selection_rate'].values[0]
sub_rates['AIR'] = sub_rates['selection_rate'] / ref_rate
sub_rates['flag'] = sub_rates['AIR'].apply(lambda x: '*** BELOW 0.80' if x < 0.80 else '')
print(sub_rates.sort_values('AIR').to_string(index=False))
worst = sub_rates.loc[sub_rates['AIR'].idxmin()]
print(f"\nWorst: {worst['subgroup']}, AIR={worst['AIR']:.3f} and "
f"n={worst['n']}")
import solas_disparity as sd
print([x for x in dir(sd) if 'impact' in x.lower() or 'ratio' in x.lower()])


This code performs an intersectional fairness analysis by combining race and sex into a single subgroup label, such as “African-American / Female,” so the model can be evaluated on intersections of identities rather than on race or sex separately. It first removes very small subgroups by keeping only those with at least 30 observations, then computes each subgroup’s selection rate for high_risk and its sample size. Using Caucasian / Male as the reference group, it calculates the adverse impact ratio (AIR) for every subgroup and flags any subgroup below the 0.80 threshold as potentially concerning. It then identifies the subgroup with the lowest AIR as the worst group and prints its AIR and sample size for interpretation. The final dir(sd) line is just exploratory: it lists Solas functions related to impact or ratio so you can see what disparity-testing tools the library provides.



Question 3 : Compute FPR and FNR disparities by race. Test statistical significance with a two-proportion z-test


The false positive rate (FPR) and false negative rate (FNR) vary substantially across racial groups.

African-American individuals have the highest FPR (0.291), meaning they are more likely to be incorrectly classified as high risk compared to other groups.
In contrast, Caucasian individuals have a much lower FPR (0.167), indicating fewer false positives.

For false negatives:

Caucasian individuals have a higher FNR (0.549), meaning they are more likely to be incorrectly classified as low risk when they actually reoffend.
African-American individuals have a lower FNR (0.342).

These results indicate a systematic disparity in error types:

African-American individuals are more likely to receive false positives (over-predicted risk) Caucasian individuals are more likely to receive false negatives (under-predicted risk)
This reflects a well-known fairness trade-off in risk models, where different groups experience different types of errors rather than equal error rates.

Question 5:
This memo evaluates potential algorithmic bias in a recidivism prediction model using established fairness metrics and statistical tests. The analysis applies Adverse Impact Ratio (AIR), Marginal Effect (ME), and Standardized Mean Difference (SMD), along with error-rate disparities including False Positive Rate (FPR) and False Negative Rate (FNR). AIR assesses relative selection rates using the EEOC 80% rule (AIR < 0.80 indicating potential disparate impact), ME captures absolute differences in outcomes, and SMD evaluates differences in continuous risk scores. All results were computed both manually and using the Solas AI disparity library to ensure consistency and reproducibility, with identical findings across methods.

The results indicate substantial disparities across race and sex. Asian individuals exhibit an AIR of 0.58, below the 0.80 threshold, while African-American individuals show the highest selection rate and AIR above 1.0. For sex, females have an AIR of 0.47 relative to males, indicating significant disparity. Intersectional analysis (race × sex) reveals more pronounced disparities, with the worst subgroup being “Other / Female” (AIR = 0.235), demonstrating that aggregate metrics can mask subgroup harms.

Error-rate analysis further reveals unequal treatment: African-American individuals have a higher FPR (0.291) than Caucasians (0.167), while Caucasians have a higher FNR (0.549). Two-proportion z-tests confirm these differences are statistically significant (p < 0.001).

This analysis has limitations. Disparities may reflect underlying base-rate differences, making it impossible to satisfy all fairness criteria simultaneously. Small subgroup sizes reduce reliability, and the metrics used identify disparities but not causal drivers such as proxy variables. Potential strategies include threshold adjustment, re-weighting, and fairness-constrained optimization; however, tradeoffs between error types require policy decisions. Overall, the findings indicate potential disparate impact requiring further regulatory review.
