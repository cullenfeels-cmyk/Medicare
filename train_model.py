import os
import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")


# ============================================================
# MEDICARE AI - MODEL TRAINING
# ============================================================

print("\n" + "=" * 60)
print("             MEDICARE AI MODEL TRAINING")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(
    DATA_DIR,
    "Training.csv"
)


# ============================================================
# 2. CHECK DATASET
# ============================================================

print("\nChecking dataset...")

if not os.path.exists(DATA_PATH):

    print("\nERROR: Training.csv was not found.")

    print("Expected location:")
    print(DATA_PATH)

    raise FileNotFoundError(
        "Training.csv not found."
    )


print("Loading Training.csv...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 3. CLEAN DATASET
# ============================================================

print("\nCleaning dataset...")


unnamed_columns = [
    col
    for col in df.columns
    if col.lower().startswith("unnamed")
]


if unnamed_columns:

    print(
        "Removing:",
        unnamed_columns
    )

    df = df.drop(
        columns=unnamed_columns
    )


if "prognosis" not in df.columns:

    raise ValueError(
        "The column 'prognosis' was not found in Training.csv."
    )


before = len(df)

df = df.drop_duplicates()

after = len(df)

print(
    "Duplicate rows removed:",
    before - after
)


# ============================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================

print("\nPreparing model data...")

X = df.drop(
    columns=["prognosis"]
)

y = df["prognosis"]


print(
    "Number of input features:",
    len(X.columns)
)

print(
    "Number of disease classes:",
    y.nunique()
)


print("\nDisease classes:")

for disease in sorted(
    y.unique()
):

    print(
        "-",
        disease
    )


# ============================================================
# 5. CONVERT FEATURES TO NUMERIC
# ============================================================

print("\nChecking feature values...")

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.fillna(0)

X = X.astype(int)


# ============================================================
# 6. SAVE FEATURE COLUMN ORDER
# ============================================================

feature_columns = list(
    X.columns
)


feature_columns_path = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)


joblib.dump(
    feature_columns,
    feature_columns_path
)


print(
    "\nCreated:",
    feature_columns_path
)

print(
    "Stored",
    len(feature_columns),
    "features."
)


# ============================================================
# 7. ENCODE DISEASE LABELS
# ============================================================

print("\nEncoding disease labels...")

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y
)


encoder_path = os.path.join(
    MODEL_DIR,
    "disease_encoder.pkl"
)


joblib.dump(
    label_encoder,
    encoder_path
)


print(
    "Created:",
    encoder_path
)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded

)


print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


# ============================================================
# 9. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest model...")

model = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    n_jobs=-1,

    class_weight="balanced"

)


model.fit(
    X_train,
    y_train
)


print(
    "Model training completed."
)


# ============================================================
# 10. EVALUATE MODEL
# ============================================================

print("\nEvaluating model...")

y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)

print(
    f"MODEL ACCURACY: {accuracy * 100:.2f}%"
)

print("=" * 60)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

report = classification_report(

    y_test,

    y_pred,

    target_names=label_encoder.classes_,

    zero_division=0

)


print(report)


# ============================================================
# 12. SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)


joblib.dump(
    model,
    model_path
)


print("\nCreated:")
print(model_path)


# ============================================================
# 13. CREATE MODEL METADATA
# ============================================================

metadata = {

    "model_type":
        "Random Forest Classifier",

    "training_dataset":
        "Training.csv",

    "training_rows":
        int(len(X_train)),

    "testing_rows":
        int(len(X_test)),

    "number_of_features":
        int(len(feature_columns)),

    "number_of_diseases":
        int(len(label_encoder.classes_)),

    "accuracy":
        float(accuracy),

    "diseases":
        list(label_encoder.classes_),

    "features":
        feature_columns

}


metadata_path = os.path.join(
    MODEL_DIR,
    "model_metadata.pkl"
)


joblib.dump(
    metadata,
    metadata_path
)


print("\nCreated:")
print(metadata_path)


# ============================================================
# 14. CREATE TREATMENT INFORMATION DATABASE
# ============================================================

print("\nCreating treatment information database...")


recommendation_database = {}


# ============================================================
# DEFAULT SAFETY INFORMATION
# ============================================================

default_warning = (
    "Do not start, stop, or change any medication without "
    "consulting a qualified healthcare professional. "
    "This system provides educational information only and "
    "is not a substitute for professional medical diagnosis."
)


default_advice = [

    "Consult a qualified healthcare professional for proper evaluation.",

    "Do not self-medicate based only on an AI prediction.",

    "Seek professional medical advice if symptoms persist or worsen."

]


# ============================================================
# CONDITION-SPECIFIC INFORMATION
# ============================================================

treatment_information = {


    "(vertigo) Paroymsal  Positional Vertigo": {

        "medicines": [
            "Treatment depends on the underlying cause and should be assessed by a healthcare professional.",
            "Specific medication or repositioning treatment should only be recommended after clinical evaluation."
        ],

        "advice": [
            "Move slowly when changing position.",
            "Avoid driving or operating machinery while experiencing significant dizziness.",
            "Consult a healthcare professional for persistent or recurrent vertigo."
        ]
    },


    "AIDS": {

        "medicines": [
            "Antiretroviral therapy is managed by qualified healthcare professionals.",
            "Treatment should be individualized based on clinical assessment and laboratory monitoring."
        ],

        "advice": [
            "Follow the treatment plan provided by an HIV specialist.",
            "Attend recommended medical and laboratory follow-ups.",
            "Seek professional care for new or worsening symptoms."
        ]
    },


    "Acne": {

        "medicines": [
            "Treatment depends on acne severity and may include professionally recommended topical or other therapies.",
            "A dermatologist can determine the appropriate treatment."
        ],

        "advice": [
            "Keep affected skin clean.",
            "Avoid squeezing or picking acne lesions.",
            "Consult a dermatologist if acne is persistent, painful, or severe."
        ]
    },


    "Alcoholic hepatitis": {

        "medicines": [
            "Treatment requires medical assessment and management of liver inflammation and its underlying cause.",
            "Medication decisions should be made by a qualified healthcare professional."
        ],

        "advice": [
            "Avoid alcohol and seek professional medical support.",
            "Follow recommended liver monitoring.",
            "Seek urgent medical attention if severe symptoms develop."
        ]
    },


    "Allergy": {

        "medicines": [
            "Treatment depends on the type and severity of the allergic reaction.",
            "A healthcare professional can recommend appropriate allergy treatment."
        ],

        "advice": [
            "Avoid known triggers when possible.",
            "Seek urgent medical attention for difficulty breathing, severe swelling, or other signs of a serious allergic reaction."
        ]
    },


    "Arthritis": {

        "medicines": [
            "Treatment depends on the type of arthritis and may include professionally recommended medicines and physical therapy."
        ],

        "advice": [
            "Maintain appropriate physical activity as advised by a healthcare professional.",
            "Discuss persistent joint pain or swelling with a healthcare professional."
        ]
    },


    "Bronchial Asthma": {

        "medicines": [
            "Asthma treatment commonly involves an individualized management plan created by a healthcare professional.",
            "Use prescribed inhalers exactly as directed by your healthcare provider."
        ],

        "advice": [
            "Identify and avoid known asthma triggers.",
            "Keep prescribed rescue medication available if applicable.",
            "Seek urgent care for severe breathing difficulty."
        ]
    },


    "Cervical spondylosis": {

        "medicines": [
            "Treatment may include physical therapy and professionally recommended pain management."
        ],

        "advice": [
            "Maintain good posture.",
            "Follow exercises recommended by a healthcare professional.",
            "Seek medical assessment for persistent neck pain or neurological symptoms."
        ]
    },


    "Chicken pox": {

        "medicines": [
            "Treatment is generally focused on symptom management and depends on age and clinical risk.",
            "A healthcare professional should determine whether specific treatment is required."
        ],

        "advice": [
            "Avoid scratching the rash.",
            "Maintain good hygiene.",
            "Avoid close contact with vulnerable individuals while contagious."
        ]
    },


    "Chronic cholestasis": {

        "medicines": [
            "Treatment depends on the underlying cause and requires medical evaluation."
        ],

        "advice": [
            "Follow up with a healthcare professional for liver and bile-related evaluation.",
            "Report worsening jaundice, abdominal symptoms, or other concerning changes."
        ]
    },


    "Common Cold": {

        "medicines": [
            "Treatment is generally supportive and depends on symptoms.",
            "A healthcare professional can advise whether any symptom-relief treatment is appropriate."
        ],

        "advice": [
            "Rest adequately.",
            "Maintain hydration.",
            "Seek medical advice if symptoms are severe or persistent."
        ]
    },


    "Dengue": {

        "medicines": [
            "Dengue requires appropriate medical monitoring, particularly when warning signs occur.",
            "Medication choices should be discussed with a healthcare professional."
        ],

        "advice": [
            "Maintain adequate hydration.",
            "Monitor symptoms carefully.",
            "Seek medical attention promptly if warning signs or worsening symptoms occur."
        ]
    },


    "Diabetes ": {

        "medicines": [
            "Diabetes treatment must be individualized by a healthcare professional.",
            "Medication and dosage depend on the patient's clinical condition."
        ],

        "advice": [
            "Monitor blood glucose as advised.",
            "Follow a balanced diet and activity plan recommended by your healthcare professional.",
            "Attend regular medical follow-ups."
        ]
    },


    "Dimorphic hemmorhoids(piles)": {

        "medicines": [
            "Treatment depends on severity and may include dietary measures and professionally recommended treatment."
        ],

        "advice": [
            "Maintain adequate dietary fiber.",
            "Drink sufficient fluids.",
            "Consult a healthcare professional for persistent bleeding or pain."
        ]
    },


    "Drug Reaction": {

        "medicines": [
            "Treatment depends on the type and severity of the reaction and requires professional assessment.",
            "Do not change prescribed medication without medical advice."
        ],

        "advice": [
            "Inform a healthcare professional about the suspected medication reaction.",
            "Seek urgent medical attention for breathing difficulty, facial swelling, or severe skin reactions."
        ]
    },


    "Fungal infection": {

        "medicines": [
            "Treatment depends on the location and type of fungal infection.",
            "A healthcare professional can recommend an appropriate antifungal treatment."
        ],

        "advice": [
            "Keep affected areas clean and dry.",
            "Avoid sharing personal items such as towels.",
            "Seek professional advice if the infection spreads or does not improve."
        ]
    },


    "GERD": {

        "medicines": [
            "Treatment may include lifestyle changes and professionally recommended acid-reducing therapy."
        ],

        "advice": [
            "Avoid foods that consistently trigger symptoms.",
            "Avoid lying down immediately after meals.",
            "Consult a healthcare professional for frequent or persistent symptoms."
        ]
    },


    "Gastroenteritis": {

        "medicines": [
            "Treatment generally focuses on hydration and management of symptoms.",
            "A healthcare professional should determine whether additional treatment is required."
        ],

        "advice": [
            "Maintain adequate fluid intake.",
            "Monitor for signs of dehydration.",
            "Seek medical care for severe or persistent symptoms."
        ]
    },


    "Heart attack": {

        "medicines": [
            "A suspected heart attack is a medical emergency and requires immediate professional treatment."
        ],

        "advice": [
            "Seek emergency medical attention immediately if symptoms suggest a heart attack.",
            "Do not rely on an AI prediction for emergency decision-making."
        ]
    },


    "Hepatitis B": {

        "medicines": [
            "Management depends on whether infection is acute or chronic and requires medical evaluation."
        ],

        "advice": [
            "Follow recommended liver monitoring.",
            "Avoid alcohol unless a healthcare professional advises otherwise.",
            "Attend specialist follow-up when recommended."
        ]
    },


    "Hepatitis C": {

        "medicines": [
            "Hepatitis C can require specialist-directed antiviral treatment.",
            "Treatment decisions should be made following medical evaluation."
        ],

        "advice": [
            "Attend recommended liver and infection monitoring.",
            "Avoid sharing items that may carry blood.",
            "Follow specialist treatment recommendations."
        ]
    },


    "Hepatitis D": {

        "medicines": [
            "Management requires evaluation by a qualified healthcare professional.",
            "Treatment depends on the clinical condition and liver status."
        ],

        "advice": [
            "Follow liver monitoring recommendations.",
            "Avoid alcohol unless medically advised otherwise.",
            "Attend specialist follow-up."
        ]
    },


    "Hepatitis E": {

        "medicines": [
            "Treatment is generally supportive, although medical assessment is important.",
            "Special circumstances may require specialist management."
        ],

        "advice": [
            "Maintain adequate hydration.",
            "Rest adequately.",
            "Seek medical advice for worsening symptoms or significant jaundice."
        ]
    },


    "Hypertension ": {

        "medicines": [
            "Blood-pressure treatment should be selected and monitored by a qualified healthcare professional."
        ],

        "advice": [
            "Monitor blood pressure as recommended.",
            "Follow a heart-healthy lifestyle.",
            "Take prescribed treatment exactly as directed."
        ]
    },


    "Hyperthyroidism": {

        "medicines": [
            "Treatment depends on the cause and may require specialist-directed therapy."
        ],

        "advice": [
            "Consult a healthcare professional for thyroid testing.",
            "Follow recommended monitoring.",
            "Report worsening palpitations or other significant symptoms."
        ]
    },


    "Hypoglycemia": {

        "medicines": [
            "Management depends on the cause and severity of low blood glucose."
        ],

        "advice": [
            "Seek immediate medical assistance if symptoms are severe or consciousness is affected.",
            "People with recurrent episodes should be medically evaluated."
        ]
    },


    "Hypothyroidism": {

        "medicines": [
            "Thyroid hormone replacement may be prescribed by a healthcare professional when clinically appropriate."
        ],

        "advice": [
            "Follow recommended thyroid blood tests.",
            "Take prescribed treatment exactly as directed.",
            "Attend regular follow-up appointments."
        ]
    },


    "Impetigo": {

        "medicines": [
            "Treatment depends on the extent of infection and should be determined by a healthcare professional."
        ],

        "advice": [
            "Keep affected skin clean.",
            "Avoid scratching or touching lesions.",
            "Avoid sharing towels and personal items."
        ]
    },


    "Jaundice": {

        "medicines": [
            "Jaundice is a symptom with many possible causes and requires medical evaluation rather than self-treatment."
        ],

        "advice": [
            "Consult a healthcare professional to determine the underlying cause.",
            "Seek prompt medical attention if jaundice is worsening or accompanied by severe symptoms."
        ]
    },


    "Malaria": {

        "medicines": [
            "Malaria requires appropriate diagnosis and professionally selected antimalarial treatment."
        ],

        "advice": [
            "Seek medical evaluation promptly.",
            "Complete any prescribed treatment exactly as directed.",
            "Seek urgent care if severe symptoms develop."
        ]
    },


    "Migraine": {

        "medicines": [
            "Migraine treatment depends on symptom severity and individual medical history."
        ],

        "advice": [
            "Rest in a quiet environment during an attack.",
            "Identify and avoid personal migraine triggers when possible.",
            "Seek professional advice for frequent or severe headaches."
        ]
    },


    "Osteoarthristis": {

        "medicines": [
            "Treatment may include physical activity, rehabilitation, and professionally recommended symptom management."
        ],

        "advice": [
            "Maintain appropriate joint-friendly activity.",
            "Discuss persistent pain or reduced mobility with a healthcare professional."
        ]
    },


    "Paralysis (brain hemorrhage)": {

        "medicines": [
            "Suspected brain hemorrhage is a medical emergency requiring immediate professional assessment and hospital treatment."
        ],

        "advice": [
            "Seek emergency medical attention immediately.",
            "Do not delay professional care because of an AI prediction."
        ]
    },


    "Peptic ulcer diseae": {

        "medicines": [
            "Treatment depends on the cause and may require professionally directed therapy."
        ],

        "advice": [
            "Consult a healthcare professional for persistent abdominal pain.",
            "Seek urgent care for vomiting blood or black stools."
        ]
    },


    "Pneumonia": {

        "medicines": [
            "Treatment depends on the cause and severity and should be determined by a healthcare professional."
        ],

        "advice": [
            "Rest and maintain adequate hydration.",
            "Seek medical evaluation for significant breathing difficulty, chest pain, or worsening symptoms."
        ]
    },


    "Psoriasis": {

        "medicines": [
            "Treatment depends on severity and may include professionally recommended topical or other therapies."
        ],

        "advice": [
            "Avoid known skin irritants.",
            "Keep skin moisturized as appropriate.",
            "Consult a dermatologist for persistent or extensive symptoms."
        ]
    },


    "Tuberculosis": {

        "medicines": [
            "Tuberculosis requires professionally supervised treatment with an appropriate treatment regimen."
        ],

        "advice": [
            "Seek medical evaluation and appropriate testing.",
            "Follow the prescribed treatment plan carefully.",
            "Attend recommended follow-up appointments."
        ]
    },


    "Typhoid": {

        "medicines": [
            "Typhoid treatment requires appropriate medical evaluation and professionally selected therapy."
        ],

        "advice": [
            "Maintain hydration.",
            "Follow the treatment plan provided by a healthcare professional.",
            "Seek medical care for persistent or worsening fever."
        ]
    },


    "Urinary tract infection": {

        "medicines": [
            "Treatment depends on the type and severity of the infection and may require professionally prescribed therapy."
        ],

        "advice": [
            "Maintain adequate hydration unless medically restricted.",
            "Consult a healthcare professional for persistent urinary symptoms.",
            "Seek prompt medical care for fever, back pain, or worsening symptoms."
        ]
    },


    "Varicose veins": {

        "medicines": [
            "Management depends on symptoms and severity and should be assessed by a healthcare professional."
        ],

        "advice": [
            "Avoid prolonged standing when possible.",
            "Regular movement may support circulation.",
            "Seek medical evaluation for pain, swelling, skin changes, or sudden worsening."
        ]
    },


    "hepatitis A": {

        "medicines": [
            "Treatment is generally supportive and should be guided by a healthcare professional."
        ],

        "advice": [
            "Rest adequately.",
            "Maintain hydration.",
            "Follow medical advice regarding liver health and follow-up."
        ]
    }
}


# ============================================================
# BUILD DATABASE FOR ALL TRAINED DISEASES
# ============================================================

for disease in label_encoder.classes_:

    # Exact match first
    info = treatment_information.get(
        disease
    )


    # Normalized match if needed
    if info is None:

        normalized_disease = (
            str(disease)
            .strip()
            .lower()
        )

        for key, value in treatment_information.items():

            normalized_key = (
                str(key)
                .strip()
                .lower()
            )

            if normalized_key == normalized_disease:

                info = value

                break


    # Use general information if a disease
    # is not explicitly listed above
    if info is None:

        info = {

            "medicines": [
                "Specific treatment depends on the individual clinical condition and should be determined by a qualified healthcare professional."
            ],

            "advice": default_advice.copy()

        }


    recommendation_database[disease] = {

        "medicines":
            info.get(
                "medicines",
                []
            ),

        "advice":
            info.get(
                "advice",
                default_advice.copy()
            ),

        "warning":
            info.get(
                "warning",
                default_warning
            )

    }


# ============================================================
# SAVE RECOMMENDATION DATABASE
# ============================================================

medicine_database_path = os.path.join(
    MODEL_DIR,
    "medicine_database.pkl"
)


joblib.dump(
    recommendation_database,
    medicine_database_path
)


print("\nCreated:")
print(
    medicine_database_path
)


# ============================================================
# VERIFY DATABASE
# ============================================================

print("\nVerifying treatment database...")

loaded_database = joblib.load(
    medicine_database_path
)


print(
    "Database type:",
    type(loaded_database)
)

print(
    "Number of diseases:",
    len(loaded_database)
)


print("\nSample treatment entries:")


for disease, information in list(
    loaded_database.items()
)[:5]:

    print(
        "\nDisease:",
        disease
    )

    print(
        "Treatment information:",
        information.get("medicines")
    )

    print(
        "Advice:",
        information.get("advice")
    )


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("             TRAINING COMPLETED")
print("=" * 60)


print("\nModels folder now contains:")


print("\n1. best_model.pkl")
print(
    "   -> Random Forest disease prediction model"
)


print("\n2. disease_encoder.pkl")
print(
    "   -> Converts disease labels"
)


print("\n3. feature_columns.pkl")
print(
    "   -> Stores the symptom features"
)


print("\n4. medicine_database.pkl")
print(
    "   -> Condition-specific treatment information"
)


print("\n5. model_metadata.pkl")
print(
    "   -> Model information and accuracy"
)


print("\n" + "=" * 60)
print("MEDICARE AI MODEL READY")
print("=" * 60)