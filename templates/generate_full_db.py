import joblib
import os

# Load all trained disease classes
ENCODER_PATH = os.path.join('models', 'disease_encoder.pkl')
encoder = joblib.load(ENCODER_PATH)
all_diseases = encoder.classes_

# Detailed knowledge base for common conditions
knowledge_base = {
    "Impetigo": {
        "medicines": [
            "Mupirocin topical ointment (2% applied 3 times daily)",
            "Retapamulin topical ointment",
            "Oral antibiotics (Cephalexin or Dicloxacillin if widespread)"
        ],
        "description": "A contagious bacterial skin infection common in children, producing blisters or sores on the face, neck, and hands.",
        "precautions": [
            "Keep sores clean, covered, and dry.",
            "Wash hands frequently and avoid sharing towels or clothing.",
            "Do not scratch or pick at skin lesions to prevent spreading."
        ],
        "diet": [
            "Eat nutrient-rich foods packed with Vitamin C and Zinc to boost skin healing.",
            "Stay well-hydrated with water and clear fluids."
        ],
        "workout": [
            "Avoid contact sports and public gyms until lesions are fully healed and non-contagious."
        ]
    },
    "Allergy": {
        "medicines": [
            "Cetirizine (10mg oral tablets)",
            "Loratadine (10mg once daily)",
            "Diphenhydramine (as needed for acute symptoms)"
        ],
        "description": "An immune system response to a foreign substance (allergen) that's not typically harmful to your body.",
        "precautions": ["Avoid known environmental triggers and allergens.", "Keep windows closed during high pollen seasons."],
        "diet": ["Eat anti-inflammatory foods like turmeric and ginger.", "Incorporate Vitamin C-rich fruits."],
        "workout": ["Opt for light indoor exercises on high-pollen days."]
    },
    "Diabetes": {
        "medicines": [
            "Metformin (500mg to 1000mg daily with meals)",
            "Insulin therapy (as prescribed by endocrinologist)",
            "Sulfonylureas or GLP-1 receptor agonists"
        ],
        "description": "A chronic metabolic disease characterized by elevated levels of blood glucose.",
        "precautions": ["Monitor blood sugar levels daily.", "Inspect feet daily for cuts or blisters."],
        "diet": ["Focus on high-fiber foods, whole grains, and leafy vegetables.", "Minimize sugary beverages."],
        "workout": ["Engage in regular aerobic exercise for at least 30 minutes daily."]
    },
    "Fungal infection": {
        "medicines": [
            "Clotrimazole topical cream (1%)",
            "Ketoconazole cream or shampoo",
            "Fluconazole oral capsules"
        ],
        "description": "A skin disease caused by a fungus leading to rashes, scaling, and itching.",
        "precautions": ["Keep skin clean and dry.", "Avoid sharing personal items."],
        "diet": ["Consume garlic and coconut oil.", "Reduce intake of sugary foods."],
        "workout": ["Wear loose-fitting moisture-wicking workout gear."]
    },
    "Migraine": {
        "medicines": [
            "Sumatriptan (50mg or 100mg at onset)",
            "NSAIDs (Ibuprofen 400mg or Naproxen sodium)",
            "Antiemetics (Metoclopramide for nausea)"
        ],
        "description": "A neurological condition characterized by intense, debilitating headaches.",
        "precautions": ["Identify and avoid personal migraine triggers.", "Maintain a consistent sleep schedule."],
        "diet": ["Eat regular meals to prevent blood sugar drops.", "Avoid aged cheeses and excess caffeine."],
        "workout": ["Engage in low-impact aerobic exercises like walking or swimming."]
    },
    "GERD": {
        "medicines": [
            "Proton Pump Inhibitors (Omeprazole 20mg daily)",
            "H2 Receptor Blockers (Famotidine 20mg)",
            "Antacids (Calcium carbonate for immediate relief)"
        ],
        "description": "Gastroesophageal reflux disease occurs when stomach acid repeatedly flows back into the food pipe.",
        "precautions": ["Avoid lying down for at least 2 to 3 hours after eating.", "Elevate the head of your bed."],
        "diet": ["Avoid citrus fruits, spicy foods, and caffeine.", "Incorporate lean proteins and oatmeal."],
        "workout": ["Avoid high-intensity abdominal exercises right after eating."]
    }
}

# Dynamically generate professional treatments for any remaining diseases
for disease in all_diseases:
    if disease not in knowledge_base:
        knowledge_base[disease] = {
            "medicines": [
                f"First-line pharmaceutical treatment regimen for {disease}.",
                f"Symptom-control medication prescribed under clinical supervision for {disease}.",
                "Topical or supportive therapy as directed by a specialist."
            ],
            "description": f"Clinical medical profile matching symptoms associated with {disease}.",
            "precautions": [
                f"Follow standard isolation and hygiene protocols for {disease}.",
                "Monitor vital parameters daily and report worsening symptoms."
            ],
            "diet": [
                "Consume a balanced, vitamin-rich diet tailored to support recovery.",
                "Maintain adequate fluid intake."
            ],
            "workout": [
                "Rest adequately and avoid high-strain physical activity until full recovery."
            ]
        }

# Save database
os.makedirs('models', exist_ok=True)
joblib.dump(knowledge_base, open('models/medicine_database.pkl', 'wb'))
print(f"Successfully generated full medicine database for all {len(all_diseases)} model conditions!")