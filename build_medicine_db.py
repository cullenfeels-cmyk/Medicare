import pickle

# Define a dictionary mapping diseases to specific medicines/treatments
medicine_database = {
    "Allergy": [
        "Cetirizine (10mg daily)",
        "Loratadine (10mg daily)",
        "Diphenhydramine (as needed for acute symptoms)"
    ],
    "Fungal infection": [
        "Clotrimazole topical cream",
        "Ketoconazole shampoo or cream",
        "Fluconazole oral tablets (if prescribed)"
    ],
    "Jaundice": [
        "IV fluids for hydration",
        "Ursodeoxycholic acid (if prescribed for itching/bile flow)",
        "Vitamin supplements (B-complex, Vitamin D, K)"
    ],
    "Drug Reaction": [
        "Discontinue the offending drug immediately",
        "Oral Antihistamines (e.g., Cetirizine)",
        "Topical corticosteroids for skin rashes"
    ],
    "Hepatitis C": [
        "Sofosbuvir / Velpatasvir (Direct-acting antivirals)",
        "Ledipasvir / Sofosbuvir",
        "Regular monitoring of liver function tests"
    ]
    # Add other conditions from your dataset here...
}

# Save the dictionary to medicine_database.pkl inside the models folder
with open('models/medicine_database.pkl', 'wb') as f:
    pickle.dump(medicine_database, f)

print("medicine_database.pkl has been successfully created and populated!")