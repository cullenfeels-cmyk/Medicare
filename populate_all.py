import joblib
import os

# Load all trained disease classes
ENCODER_PATH = os.path.join('models', 'disease_encoder.pkl')
encoder = joblib.load(ENCODER_PATH)
all_diseases = encoder.classes_

# Comprehensive Healthcare Knowledge Base
knowledge_base = {
    "Allergy": {
        "medicines": [
            "Cetirizine (10mg oral tablets)",
            "Loratadine (10mg once daily)",
            "Diphenhydramine (as needed for acute symptoms)"
        ],
        "description": "An immune system response to a foreign substance (allergen) that's not typically harmful to your body.",
        "precautions": [
            "Avoid known environmental triggers and allergens.",
            "Keep windows closed during high pollen seasons.",
            "Wear a mask when cleaning or doing outdoor work."
        ],
        "diet": [
            "Eat anti-inflammatory foods like turmeric, ginger, and local honey.",
            "Incorporate Vitamin C-rich fruits (citrus, berries) to support immunity.",
            "Avoid processed foods and artificial additives."
        ],
        "workout": [
            "Opt for light indoor exercises on high-pollen days.",
            "Practice gentle yoga and deep breathing exercises.",
            "Avoid intense outdoor cardio during allergy seasons."
        ]
    },
    "Fungal infection": {
        "medicines": [
            "Clotrimazole topical cream (1%)",
            "Ketoconazole cream or shampoo",
            "Fluconazole oral capsules"
        ],
        "description": "A skin disease caused by a fungus that can lead to rashes, irritation, scaling, and itching.",
        "precautions": [
            "Keep your skin clean and dry, especially in skin folds.",
            "Avoid sharing towels, clothing, or personal care items.",
            "Wear breathable cotton clothing."
        ],
        "diet": [
            "Consume garlic and coconut oil, known for natural antifungal properties.",
            "Reduce intake of sugary foods and refined carbohydrates which feed fungi.",
            "Stay well-hydrated with water and green tea."
        ],
        "workout": [
            "Wear loose-fitting moisture-wicking workout gear.",
            "Shower immediately after exercising and dry thoroughly.",
            "Sanitize gym equipment before and after use."
        ]
    },
    "Acne": {
        "medicines": [
            "Benzoyl Peroxide topical gel (2.5% to 5%)",
            "Salicylic Acid facial cleanser",
            "Topical Tretinoin or Clindamycin gel"
        ],
        "description": "A skin condition that occurs when hair follicles become plugged with oil and dead skin cells.",
        "precautions": [
            "Wash your face twice daily with a gentle cleanser.",
            "Avoid touching your face or picking at blemishes.",
            "Use non-comedogenic (won't clog pores) skincare and makeup products."
        ],
        "diet": [
            "Focus on low-glycemic foods (whole grains, vegetables, fresh fruits).",
            "Increase intake of Omega-3 fatty acids (walnuts, flaxseeds, fish).",
            "Limit dairy products and processed sugars if they trigger breakouts."
        ],
        "workout": [
            "Wash your face right after sweating heavily.",
            "Avoid wearing tight headbands or helmets that trap sweat on the skin.",
            "Wear clean workout clothes to prevent body acne."
        ]
    },
    "Migraine": {
        "medicines": [
            "Sumatriptan (50mg or 100mg at onset)",
            "NSAIDs (Ibuprofen 400mg or Naproxen sodium)",
            "Antiemetics (Metoclopramide for nausea)"
        ],
        "description": "A neurological condition characterized by intense, debilitating headaches often accompanied by nausea and sensitivity to light and sound.",
        "precautions": [
            "Identify and avoid personal migraine triggers (stress, loud noises, bright lights).",
            "Maintain a consistent sleep and meal schedule.",
            "Stay hydrated throughout the day."
        ],
        "diet": [
            "Eat regular, balanced meals to prevent blood sugar drops.",
            "Avoid common triggers like aged cheeses, artificial sweeteners, and excess caffeine.",
            "Include magnesium-rich foods like leafy greens, nuts, and seeds."
        ],
        "workout": [
            "Engage in low-impact aerobic exercises like walking, swimming, or cycling.",
            "Always include a proper warm-up to prevent exercise-induced tension headaches.",
            "Practice neck and shoulder stretches regularly."
        ]
    },
    "GERD": {
        "medicines": [
            "Proton Pump Inhibitors (Omeprazole 20mg daily)",
            "H2 Receptor Blockers (Famotidine 20mg)",
            "Antacids (Calcium carbonate for immediate relief)"
        ],
        "description": "Gastroesophageal reflux disease (GERD) occurs when stomach acid repeatedly flows back into the tube connecting your mouth and stomach.",
        "precautions": [
            "Avoid lying down for at least 2 to 3 hours after eating.",
            "Elevate the head of your bed if nighttime reflux is an issue.",
            "Eat smaller, more frequent meals instead of large portions."
        ],
        "diet": [
            "Avoid trigger foods like citrus fruits, tomatoes, spicy foods, chocolate, and caffeine.",
            "Incorporate lean proteins, oatmeal, and non-citrus vegetables.",
            "Drink plenty of water between meals rather than large amounts during meals."
        ],
        "workout": [
            "Avoid high-intensity abdominal exercises or heavy lifting right after eating.",
            "Choose upright activities like walking or stationary cycling over floor-based poses that compress the stomach.",
            "Maintain a healthy weight to reduce pressure on the abdomen."
        ]
    }
}

# Dynamically populate defaults for any remaining diseases in the model
for disease in all_diseases:
    if disease not in knowledge_base:
        knowledge_base[disease] = {
            "medicines": [
                f"Targeted pharmacological treatment prescribed for {disease}.",
                "Symptom-relief support as advised by a physician."
            ],
            "description": f"Clinical condition associated with the analyzed symptom profile: {disease}.",
            "precautions": [
                f"Monitor your symptoms closely and maintain records for your doctor.",
                "Avoid over-exertion and prioritize adequate rest."
            ],
            "diet": [
                "Follow a balanced, nutrient-dense diet rich in vitamins and minerals.",
                "Ensure proper daily hydration with water and electrolyte-rich fluids."
            ],
            "workout": [
                "Engage in light physical activity only as tolerated.",
                "Avoid strenuous workouts until symptoms completely resolve."
            ]
        }

# Save as the updated database file
os.makedirs('models', exist_ok=True)
with open('models/medicine_database.pkl', 'wb') as f:
    joblib.dump(knowledge_base, f)

print(f"Successfully updated medicine_database.pkl with descriptions, precautions, diet, and workout data for all {len(all_diseases)} conditions!")