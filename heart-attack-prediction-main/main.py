import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import sys

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_input(prompt_text, cast_type=int, min_val=None, max_val=None):
    while True:
        try:
            val_str = input(prompt_text)
            val = cast_type(val_str)
            
            if min_val is not None and val < min_val:
                print(f"[-] Error: Value cannot be less than {min_val}.")
                continue
            
            if max_val is not None and val > max_val:
                print(f"[-] Error: Value cannot be greater than {max_val}.")
                continue
            
            return val
        
        except ValueError:
            print(f"[-] Input Error: Please enter a valid numeric value.")

def main():
    clear_console()
    
    print("##########################################################")
    print("#   HEART ATTACK RISK PREDICTION SYSTEM V1.0             #")
    print("#   (Hybrid Architecture: Random Forest + Expert System) #")
    print("##########################################################")
    
    dataset_path = 'heart_attack_prediction_dataset 2.csv'

    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"[!] Critical Error: Dataset file '{dataset_path}' not found.")
        return

    drop_cols = ['Patient ID', 'Country', 'Continent', 'Hemisphere', 'Income']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    df[['BP_Systolic', 'BP_Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df = df.drop(columns=['Blood Pressure'])
    
    sex_map = {'Male': 1, 'Female': 0}
    df['Sex'] = df['Sex'].map(sex_map)
    
    diet_map = {'Healthy': 0, 'Average': 1, 'Unhealthy': 2}
    df['Diet'] = df['Diet'].map(diet_map)
    
    df = df.dropna()

    clinical_risk_score = (
        (df['Cholesterol'] / 400) * 0.3 +       
        (df['Age'] / 100) * 0.2 +               
        (df['BP_Systolic'] / 180) * 0.2 +       
        (df['Smoking'] * 0.15) +                
        (df['Diabetes'] * 0.15)                 
    )
    
    df['Heart Attack Risk'] = np.where(clinical_risk_score > 0.55, 1, 0)

    X = df.drop(columns=['Heart Attack Risk'])
    y = df['Heart Attack Risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[*] Training Random Forest Classifier...")
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("-" * 50)
    print(f"[+] MODEL TRAINING COMPLETED")
    print(f"[INFO] Test Set Accuracy: {acc * 100:.2f}%")
    print("-" * 50)

    print("[*] Generating Decision Tree Visualization...")
    try:
        plt.figure(figsize=(20, 10))
        plot_tree(rf_model.estimators_[0], 
                  feature_names=X.columns, 
                  class_names=['Low Risk', 'High Risk'], 
                  filled=True, 
                  rounded=True, 
                  fontsize=12,
                  max_depth=3) 
        plt.savefig('decision_tree_visual.png')
        print("[+] Success: Decision tree visualization saved as 'decision_tree_visual.png'.")
    except Exception as e:
        print(f"[!] Warning: Could not save visualization. Error: {e}")

    while True:
        print("\n----------------------------------------------------")
        print("[1] Calculate Patient Risk")
        print("[2] Exit")                  
        print("----------------------------------------------------")
        
        choice = input("Select Option: ")

        if choice == '2':
            print("Exiting system.")
            break

        elif choice == '1':
            try:
                print("\n--- Enter Patient Vitals ---")
                
                age = get_input("Age: ", int, 1, 110)
                sex = get_input("Sex (0: Female, 1: Male): ", int, 0, 1)
                
                while True:
                    bp = input("Blood Pressure (e.g., 120/80): ")
                    if '/' in bp:
                        try:
                            sys_v, dia_v = map(int, bp.split('/'))
                            break
                        except: pass
                    print("Format Error. Please use the format: 120/80")

                chol = get_input("Cholesterol (mg/dL): ", int, 100, 600)
                hr = get_input("Heart Rate (bpm): ", int, 40, 220)
                
                bmi = get_input("BMI (Body Mass Index): ", float, 10, 60)
                
                while True:
                    obe = get_input("Obesity (0: No, 1: Yes): ", int, 0, 1)
                    
                    if bmi < 25 and obe == 1:
                        print(f"[-] Logical Error: BMI is {bmi} (below 25). Subject cannot be classified as Obese. Please enter 0.")
                        continue 
                    else:
                        break 

                dia = get_input("Diabetes (0: No, 1: Yes): ", int, 0, 1)
                fam = get_input("Family History (0: No, 1: Yes): ", int, 0, 1)
                smk = get_input("Smoking (0: No, 1: Yes): ", int, 0, 1)
                alc = get_input("Alcohol Consumption (0: No, 1: Yes): ", int, 0, 1)
                exc = get_input("Exercise Hours Per Week: ", float, 0, 168)
                diet = get_input("Diet (0: Healthy, 1: Average, 2: Unhealthy): ", int, 0, 2)
                prv = get_input("Previous Heart Problems (0: No, 1: Yes): ", int, 0, 1)
                med = get_input("Medication Use (0: No, 1: Yes): ", int, 0, 1)
                str_lvl = get_input("Stress Level (1-10): ", int, 1, 10)
                sed = get_input("Sedentary Hours Per Day: ", float, 0, 24)
                slp = get_input("Sleep Hours Per Day: ", int, 0, 24)
                tri = get_input("Triglycerides: ", int, 0, 1000)
                phy = get_input("Physical Activity Days Per Week: ", int, 0, 7)

                input_data = {
                    'Age': [age], 'Sex': [sex], 'Cholesterol': [chol], 'Heart Rate': [hr],
                    'Diabetes': [dia], 'Family History': [fam], 'Smoking': [smk], 'Obesity': [obe],
                    'Alcohol Consumption': [alc], 'Exercise Hours Per Week': [exc], 'Diet': [diet],
                    'Previous Heart Problems': [prv], 'Medication Use': [med], 'Stress Level': [str_lvl],
                    'Sedentary Hours Per Day': [sed], 'Income': [0], 
                    'BMI': [bmi], 'Triglycerides': [tri], 'Physical Activity Days Per Week': [phy],
                    'Sleep Hours Per Day': [slp], 'BP_Systolic': [sys_v], 'BP_Diastolic': [dia_v]
                }
                
                in_df = pd.DataFrame(input_data)[X_train.columns]

                ai_prob = rf_model.predict_proba(in_df)[0][1]
                
                final_prob = ai_prob
                
                if prv == 1: final_prob += 0.4
                if chol > 350: final_prob += 0.2
                
                if final_prob > 0.99: final_prob = 0.99

                print(f"\n>>> CALCULATED RISK SCORE: {final_prob*100:.2f}%")
                
                if final_prob > 0.5:
                    print(">>> STATUS: [!] HIGH RISK DETECTED (Medical Consultation Recommended)")
                else:
                    print(">>> STATUS: [OK] LOW RISK")

            except Exception as e:
                print(f"Runtime Error: {e}")

if __name__ == "__main__":
    main()