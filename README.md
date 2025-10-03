#  Predictive Maintenance for Pumps (Shell Use-Case)

##  Problem

In oil & gas plants, pumps operate **24/7**.
If a pump fails unexpectedly, **production stops** → costing millions per day in downtime.

- The goal of this project is to **predict pump failures in advance** using sensor data, enabling Shell (or any industry) to schedule **preventive maintenance** before breakdown.

---

##  Dataset

* **Source**: [Kaggle Pump Sensor Dataset](https://www.kaggle.com/nphantawee/pump-sensor-data)
* **Features**:

  * `temperature`
  * `vibration`
  * `pressure`
  * `flow_rate`
  * `machine_status` (Normal / Warning / Failure)

We convert `machine_status` into a binary target:

* `0 = Healthy`
* `1 = Failure`

---

##  Workflow

### 1. Data Preprocessing

* Convert categorical labels → binary failure column
* Drop unused fields (e.g., timestamp)

### 2. Exploratory Data Analysis (EDA)

* Visualize differences in **temperature, vibration, pressure** between healthy & failing pumps
* Identify **patterns before failure** (e.g., abnormal vibration spikes)

### 3. Feature Engineering

* Rolling averages (5-reading mean)
* Lag features (previous reading values)
* Trend features (moving statistics)

### 4. Model Training

* Algorithm: **Random Forest Classifier**
* Split: 80% training, 20% testing
* Evaluation: **Precision, Recall, F1-score, Accuracy**

### 5. Results

Example output:

```
Accuracy: ~94%
Failure detection Recall: ~0.85
```

Confusion matrix shows strong ability to detect failing pumps.

### 6. Model Saving

* Save trained model → `pump_model.pkl` using **pickle**
* Can be re-used without retraining.

### 7. Demo App (Optional)

* Built with **Streamlit**
* User inputs (`temperature, vibration, pressure, flow_rate`)
* Model predicts:

  *  Pump is Healthy
  *  Pump Failure Risk

---

##  Business Impact for Shell

* **Unplanned downtime = millions lost/day** in refinery & offshore operations
* Predictive maintenance → reduces unplanned outages
* Our model (90% accuracy) helps Shell:

  * Detect failures early
  * Schedule maintenance efficiently
  * Save **$X million annually** in avoided downtime

---

##  How to Run

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### Jupyter Notebook

Run the notebook end-to-end:

```bash
jupyter notebook Predictive_Maintenance_Pumps.ipynb
```



---

##  Repository Structure

```
predictive-maintenance-pump/
│── notebooks/            # EDA + Model Training (ipynb)
│── pump_model.pkl        # Saved model
│── README.md             # Project documentation
```
---

##  Author

* **Om Panchal**
*  *Aspiring  Data Scientist*
