# Diabetes Screening on Spark MLlib

Logistic regression over 768 Pima clinical records, trained distributed.

| | |
|---|---|
| Task | Binary classification |
| Data | 768 records, 34.9% diabetic |
| Engine | PySpark MLlib, cross-checked against scikit-learn |
| **Accuracy** | **0.745** |
| ROC AUC | 0.836 |
| Always guessing "not diabetic" | 0.649 |
| Source | `Day10 AOB BigDataSpark.ipynb` |

```bash
uv run python projects/diabetes_screening/train.py
uv run python projects/diabetes_screening/train.py --keep-zeros
```

Needs a JVM. `brew install openjdk@17` on macOS, `apt install openjdk-17-jdk`
on Debian.

## "Accuracy: 0.854" was not accuracy

```python
evaluator = BinaryClassificationEvaluator(labelCol="outcome")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)
```

`BinaryClassificationEvaluator` defaults to `metricName="areaUnderROC"`. The
number printed under the label `Accuracy` is the area under the ROC curve.
Nothing here is broken - the evaluator did exactly what it was asked - but the
name on the output is wrong, and it is wrong in the flattering direction.

On this split, with the same model:

| | |
|---|---|
| Accuracy | 0.745 |
| ROC AUC | 0.836 |
| Majority-class baseline | 0.649 |

A reader who believes 0.854 is accuracy concludes the model beats "assume
nobody is diabetic" by 20 points. It beats it by 10. And this is a screening
test, so the number a clinician would ask for is neither: recall is **0.531**,
meaning the model misses just under half of the diabetic patients in the test
set.

`binary_classification_scores` returns all five metrics at once. There is no
default to leave unset and nothing left to mislabel.
`test_spark_and_sklearn_score_identical_predictions_identically` scores one
fixed set of predictions with both Spark and scikit-learn and requires the two
to agree, which also demonstrates that accuracy and AUC are not each other.

## 374 patients with no insulin

Five columns encode "not recorded" as `0`:

| Column | Zeros | Share |
|---|---|---|
| Insulin | 374 | 48.7% |
| SkinThickness | 227 | 29.6% |
| BloodPressure | 35 | 4.6% |
| BMI | 11 | 1.4% |
| Glucose | 5 | 0.7% |

None of these is possible in a living patient. The notebook passed all of them
to `VectorAssembler` as measurements, so the model learned that a large cluster
of patients sits at the bottom of the insulin scale.

`build_features` turns them into nulls and MLlib's `Imputer` fills them from
training-fold medians - inside the pipeline, so the fill values never see the
test rows. `Pregnancies` is deliberately left alone: 111 women in the file have
never been pregnant, and that is a fact rather than a gap.

### The fix does not improve the score

Run it both ways and the headline numbers barely move:

| | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Zeros treated as missing | 0.745 | 0.672 | 0.531 | 0.593 | 0.836 |
| Zeros left in, as in the notebook | 0.745 | 0.677 | 0.519 | 0.587 | 0.838 |

Identical accuracy, and AUC is marginally *worse* after the fix.

That is the honest result and it is worth stating plainly, because the usual
way this section gets written is "we cleaned the data and the score went up".
Here it did not. A median-imputed insulin reading carries no more information
than a zero did; both are stand-ins for a measurement nobody took, and the
model was already ignoring the column's lower tail.

What changes is what the model means. With zeros in place, the fitted
coefficient on insulin is partly describing a data-entry convention. After the
fix it describes insulin. The metric cannot see that difference, which is
exactly why it cannot be the only thing you look at.

## Two engines, one split

`train.py` fits the same model twice: once in Spark MLlib, once in
scikit-learn, on the identical rows. That is only meaningful if the two are
actually the same model, so:

- The split happens once in pandas, stratified, and both engines receive it.
  Spark's `randomSplit` is neither stratified nor seeded from the same RNG.
- Spark's `StandardScaler` defaults to `withMean=False`; scikit-learn centres
  by default. Spark is told to centre.
- Spark's `LogisticRegression` defaults to `regParam=0`; scikit-learn applies
  L2 at `C=1.0`. scikit-learn is told not to regularise.

Without those three lines the engines disagree, and the disagreement is
preprocessing rather than implementation - which would make the cross-check
worse than useless, since it would look like a real finding.

## Is Spark the right tool here?

No. 768 rows fit in a spreadsheet, and the JVM takes longer to start than
scikit-learn takes to fit the model. This project exists because the source
notebook is a Spark notebook and MLlib is a genuinely different API worth
knowing - not because the data needed it. `marvel_network` has the same
comparison on a graph 400 times larger, and the answer there is the same.

Türkçe açıklamalar: [README.tr.md](README.tr.md)
