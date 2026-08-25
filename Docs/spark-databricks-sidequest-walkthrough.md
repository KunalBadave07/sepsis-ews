# Side-Quest — PySpark Batch Feature Engineering on Databricks
### Rebuilding `build_dataset.py` at scale. Every click, every command.

**What this is, honestly:** your 238,010-row dataset runs fine on pandas in a few minutes — you don't *need* Spark for data this size. This side-quest exists to prove you can operate Spark/Databricks correctly, as a genuine, separate skill line for your resume, not because the core project requires it. Keep that framing in your own head and in how you describe this to anyone who asks — overselling "I needed Spark for this" to someone who knows the data volume is an easy claim to get caught on.

---

## Step 1 — Create a free Databricks Community Edition account
1. Go to `https://www.databricks.com/try-databricks`
2. Look for the **Community Edition** option — it's usually a smaller link near the bottom of the signup form, separate from the "start free trial" enterprise flow. If you don't see it immediately, go directly to `https://community.cloud.databricks.com/login.html` and click "Sign Up."
3. Fill in your details, verify your email.
4. **Confirm you're in Community Edition, not a 14-day trial** — Community Edition is permanently free (a single small cluster, no credit card, no expiration) but it's easy to accidentally sign up for the trial instead, which *does* ask for a card eventually. If the signup flow asks for payment info at any point, stop and go back — that's the wrong one.

## Step 2 — Create a cluster
1. In the Databricks workspace, click **Compute** in the left sidebar.
2. Click **Create Compute**.
3. Community Edition auto-configures a small single-node cluster for you — accept the defaults, give it a name like `sepsis-spark`, click **Create**.
4. Wait for the green "Running" status before continuing — this can take a couple minutes.

## Step 3 — Create a new notebook
1. Click **Workspace** in the left sidebar → your username → **Create** → **Notebook**.
2. Name it `sepsis_spark_features`.
3. Set the language dropdown to **Python**.
4. Attach it to the `sepsis-spark` cluster from Step 2 (dropdown at the top of the notebook).

## Step 4 — Download the raw data directly into Databricks
The PhysioNet 2019 Challenge dataset doesn't require credentialed login (unlike MIMIC-IV), so you can pull it straight from a notebook cell. Type this into the first cell and run it (Shift+Enter):
```python
%sh
wget -q https://physionet.org/files/challenge-2019/1.0.0/training/training_setA.zip -O /tmp/training_setA.zip
unzip -q /tmp/training_setA.zip -d /tmp/training_setA
ls /tmp/training_setA | head -5
```
You should see a list of `.psv` filenames printed. If `wget` fails, double-check the URL still resolves in a browser first — PhysioNet occasionally reorganizes file paths between dataset versions.

New cell — move the files into DBFS (Databricks' distributed file system, so Spark across the cluster can see them, not just the driver node):
```python
dbutils.fs.cp("file:/tmp/training_setA", "dbfs:/FileStore/sepsis/training_setA", recurse=True)
display(dbutils.fs.ls("dbfs:/FileStore/sepsis/training_setA")[:5])
```

## Step 5 — Read all patient files into one Spark DataFrame
New cell:
```python
from pyspark.sql import functions as F

raw_df = (
    spark.read
    .option("header", True)
    .option("sep", "|")
    .option("nullValue", "NaN")
    .csv("dbfs:/FileStore/sepsis/training_setA/*.psv")
    .withColumn("source_file", F.input_file_name())
    .withColumn("patient_id", F.regexp_extract("source_file", r"(p\d+)\.psv", 1))
)

print("Total raw rows:", raw_df.count())
print("Distinct patients:", raw_df.select("patient_id").distinct().count())
```
**What you're checking:** the patient count here should be in the same ballpark as what you saw locally in Sprint 2 (roughly 19,000-20,000 for this file set — Spark reading every `.psv` file in the directory and correctly extracting a `patient_id` from each filename via regex is the thing most likely to silently go wrong here, so verify this number before moving on).

## Step 6 — Compute the SAME rolling features, the Spark way
This is the core of the exercise — the same causal, per-patient, backward-looking rolling window logic from `build_dataset.py`, expressed as a Spark `Window` function instead of a pandas `.rolling()` call.

New cell:
```python
from pyspark.sql.window import Window

typed_df = (
    raw_df
    .withColumn("ICULOS", F.col("ICULOS").cast("int"))
    .withColumn("heart_rate", F.col("HR").cast("double"))
    .withColumn("resp_rate", F.col("Resp").cast("double"))
    .withColumn("sbp", F.col("SBP").cast("double"))
    .withColumn("map_bp", F.col("MAP").cast("double"))
    .withColumn("temp_c", F.col("Temp").cast("double"))
    .withColumn("spo2", F.col("O2Sat").cast("double"))
    .withColumn("wbc", F.col("WBC").cast("double"))
    .withColumn("lactate", F.col("Lactate").cast("double"))
    .withColumn("SepsisLabel", F.col("SepsisLabel").cast("int"))
)

# rowsBetween(-7, 0) = "this row plus the 7 before it" — an 8-row trailing
# window, same as pandas' window=8 in Sprint 2. ORDER BY ICULOS is what
# makes this causal: it can only look backward, never forward, per patient.
window_spec = (
    Window.partitionBy("patient_id")
    .orderBy("ICULOS")
    .rowsBetween(-7, 0)
)

features_df = (
    typed_df
    .withColumn("hr_rolling_mean", F.avg("heart_rate").over(window_spec))
    .withColumn("hr_rolling_std", F.coalesce(F.stddev("heart_rate").over(window_spec), F.lit(0.0)))
    .withColumn("map_rolling_mean", F.avg("map_bp").over(window_spec))
    .withColumn("map_rolling_std", F.coalesce(F.stddev("map_bp").over(window_spec), F.lit(0.0)))
    .withColumn("shock_index", F.col("heart_rate") / F.col("sbp"))
)
```

**Why `F.coalesce(..., F.lit(0.0))` on the std columns:** the very first row in any patient's window has no variance to compute yet (only one data point), so Spark's `stddev` returns `null` there — same edge case pandas' `.std().fillna(0)` handled in Sprint 2. Miss this and you'll get nulls propagating downstream instead of zeros.

## Step 7 — Clean up and select final columns
```python
FINAL_COLS = [
    "patient_id", "ICULOS", "heart_rate", "resp_rate", "sbp", "map_bp",
    "temp_c", "spo2", "wbc", "lactate", "hr_rolling_mean", "hr_rolling_std",
    "map_rolling_mean", "map_rolling_std", "shock_index", "SepsisLabel",
]

clean_df = (
    features_df
    .select(*FINAL_COLS)
    .na.drop(subset=["heart_rate", "resp_rate", "sbp", "map_bp", "temp_c", "spo2"])
)

clean_df.show(5)
```

## Step 8 — THE parity check — does this match your pandas pipeline?
This is the step that actually matters. Anyone can write Spark code that runs without erroring — the real proof is that it produces the *same result* as the pipeline you already validated in Sprint 2.

New cell:
```python
total_rows = clean_df.count()
distinct_patients = clean_df.select("patient_id").distinct().count()
prevalence = clean_df.agg(F.avg("SepsisLabel")).collect()[0][0]

print(f"Spark pipeline — total rows: {total_rows}")
print(f"Spark pipeline — distinct patients: {distinct_patients}")
print(f"Spark pipeline — sepsis prevalence: {prevalence:.4%}")
```
Compare directly against your known pandas numbers from Sprint 2/4: **238,010 rows, 19,559 patients, 1.95% prevalence.** They won't be pixel-identical (Spark's `stddev` uses a slightly different default formula than pandas' in edge cases, and file read order can affect nothing here since we're aggregating, not row-ordering) — but they should be **close**, within a small percentage. If Spark reports something wildly different — say, half the row count, or a prevalence off by 2x — don't shrug it off as "Spark does things differently." Go back and check the regex in Step 5 (a patient_id extraction bug would silently drop or merge patients) before trusting this pipeline.

## Step 9 — Write the output as a Delta table
New cell:
```python
clean_df.write.format("delta").mode("overwrite").saveAsTable("sepsis_features_spark")
```
Delta Lake is Databricks' default table format — versioned, ACID-compliant storage on top of Parquet. Using it here (instead of just writing a plain Parquet file) is itself a small, legitimate resume detail: "wrote validated features to a Delta Lake table" is a real, specific claim, not filler.

Confirm it landed:
```python
%sql
SELECT COUNT(*) FROM sepsis_features_spark
```

---

## What to actually say about this on a resume
Be precise, not grandiose:
> *"Implemented an alternative PySpark batch feature-engineering pipeline on Databricks, replicating the project's causal rolling-window feature logic (window functions, partitioned by patient) and validating output parity against the primary pandas pipeline; wrote results to a Delta Lake table."*

That's a real, specific, checkable claim. It doesn't say "built the production pipeline in Spark" (it isn't — pandas/Polars still run the real system), and it doesn't oversell scale you didn't actually need. It says exactly what you did: you can operate distributed data engineering tools correctly and validate their output against a known-correct baseline. That's the actual skill being demonstrated, and it's a real one.

---

## Side-Quest — Definition of Done
- [ ] Databricks Community Edition account created — free tier confirmed, no card on file
- [ ] Raw PhysioNet data loaded into Spark via a notebook, patient count sanity-checked
- [ ] Rolling features computed via `Window` functions, correctly causal (ordered by `ICULOS`, backward-only)
- [ ] **Parity check passes** — row count, patient count, and prevalence closely match the original pandas pipeline
- [ ] Output written to a Delta table
- [ ] You can explain, out loud, why this wasn't strictly necessary for a dataset this size, and what it does demonstrate instead

When that's done, the whole project — Sprints 1 through 4, the SWADT paper, and this side-quest — is a genuinely complete, honestly-documented portfolio. From here it's really just Sprint 5-that-isn't-a-sprint: the portfolio wrap-up (resume bullets, LinkedIn post, deciding whether to run the real SWADT validation protocol) whenever you're ready for it.
