import streamlit as st
import joblib
import pandas as pd
import numpy as np
from utils import calculate_len_texts, ciudad_a_poblacion, raie_if_not_colombia, translate_date, translate_time
import clean_class as cl

# Load the trained model and options
model = joblib.load('best_random_forest_model.pkl')
options = joblib.load('options.pkl')
ciudades = pd.read_excel("ciudades.xlsx")

# Define feature categories
company_controlled_features = [
    'vacant_min_salary', 'vacant_experience_months_number', 'vacant_education_level_name',
    'vacant_is_urgent', 'vacant_disabled_people', 'vacant_is_remote',
    'vacant_job_experience_is_required', 'vacant_cities_is_required',
    'vacant_salary_is_required', 'vacant_education_levels_is_required'
]

market_controlled_features = ['suggested_candidates_mean_affinity', 'poblacion']

# Define a function to make predictions
def predict(input_data):
    prediction = model.predict(input_data)
    return prediction

# Define a function to recommend changes based on feature importances
def recommend_changes(input_data, model):
    estimators = model.named_steps['random_forest'].estimators_
    feature_names = options["columns"]
    recommendations = {}
    votes = [estimator.predict(input_data)[0] for estimator in estimators]
    vote_counts = pd.Series(votes).value_counts().to_dict()
    # filter estimators to show only where votes = 0
    estimators = [estimator for estimator, vote in zip(estimators, votes) if vote == 0]
    

    entry = 0
    for estimator in estimators:
        node_indicator = estimator.decision_path(input_data)
        
        # Get the feature and threshold for each decision node
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        impurity = estimator.tree_.impurity
        n_node_samples = estimator.tree_.n_node_samples
        value = estimator.tree_.value

        # Prioritize the first few nodes
        for node in range(node_indicator.shape[1]):
            if node_indicator[0, node] and feature[node] != -2:  # Check if the node is a split node
                feature_name = feature_names[feature[node]]
                threshold_value = threshold[node]
                feature_value = input_data.iloc[0, feature[node]]
                
                # Calculate the Gini impurity change for the left and right children
                left_child = estimator.tree_.children_left[node]
                right_child = estimator.tree_.children_right[node]
                
                gini_left = impurity[left_child] * n_node_samples[left_child]
                gini_right = impurity[right_child] * n_node_samples[right_child]
                gini_change = impurity[node] * n_node_samples[node] - gini_left - gini_right
                
                class_of_interest = 0  # Assuming class 0 is non-critical
                if (value[left_child][0][class_of_interest] > value[right_child][0][class_of_interest]):
                    preferred_child = left_child
                else:
                    preferred_child = right_child
                # Check if the preferred node is already in the path
                if node_indicator[0, preferred_child]:
                    continue  # No recommendation needed if the preferred node is already in the path
                
                direction = "increase" if threshold_value > input_data.iloc[0, feature[node]] else "decrease"
                



                recommendations[entry] = {
                    "feature_name":feature_name,
                    "feature_value": feature_value,
                    "threshold_value": threshold_value,
                    "direction": direction,
                    "gini_change": gini_change,
                }

                entry += 1

    return recommendations, vote_counts

# Define the Streamlit app
st.title("Predicción de Vacantes Críticas")

st.write("""
### Ingrese los datos de la vacante para predecir si es crítica o no
""")

# Create input fields for user data
input_data = {
    'vacant_experience_and_positions': st.selectbox('Experiencia y posición de la vacante', options["experiencias"]),
    'len_texts': calculate_len_texts(st.text_area('Descripción de la vacante')),
    'date': st.date_input('Fecha de publicación'),
    'time_of_day': translate_time(st.time_input('Hora de la vacante')),
    'vacant_min_salary': st.number_input('Salario mínimo de la vacante', min_value=500000, max_value=30000000, step=100000),
    'poblacion': ciudad_a_poblacion(st.selectbox('Ciudad', ciudades["DPMP"]), ciudades),
    'vacant_country_name': raie_if_not_colombia(st.selectbox("País", ["Colombia", "México", "Perú"])),
    'vacant_education_level_name': st.selectbox('Nivel educativo requerido', options["educacion"]),
    'vacant_is_urgent': st.checkbox('¿Es una vacante urgente?'),
    'vacant_disabled_people': st.checkbox('¿Admite personas con discapacidad?'),
    'vacant_experience_months_number': st.number_input('Número de meses de experiencia requeridos', min_value=0, step=1),
    'vacant_is_remote': st.checkbox('¿Es una vacante remota?'),
    'vacant_job_experience_is_required': st.checkbox('¿Es obligatoria la experiencia laboral?'),
    'vacant_cities_is_required': st.checkbox('¿Es obligatoria la ciudad?'),
    'vacant_salary_is_required': st.checkbox('¿Es obligatorio el salario?'),
    'vacant_education_levels_is_required': st.checkbox('¿Es obligatorio el nivel educativo?'),
}

# Function to generate random suggested_candidates_mean_affinity
def generate_random_affinity():
    return np.random.uniform(0.65, 0.76)

# Initialize or update session state for suggested_candidates_mean_affinity
if 'suggested_candidates_mean_affinity' not in st.session_state:
    st.session_state.suggested_candidates_mean_affinity = generate_random_affinity()

if st.button('Generar afinidad aleatoria'):
    st.session_state.suggested_candidates_mean_affinity = generate_random_affinity()

st.write(f"Afinidad con la bolsa de candidatos (generado aleatoriamente): {st.session_state.suggested_candidates_mean_affinity:.2f}")

input_data["suggested_candidates_mean_affinity"] = st.session_state.suggested_candidates_mean_affinity
input_data["week_day"], input_data["day"] = translate_date(input_data["date"])
input_data.pop("date")
input_df = pd.DataFrame([input_data])
input_df = cl.NewDataCleaner().transform(X=input_df)

# Ensure all columns are present in the input data
cols = options["columns"]
for c in cols:
    if c not in input_df.columns:
        input_df[c] = False

input_df = input_df[cols]

# Make prediction
if st.button('Predecir'):
    prediction = predict(input_df)
    st.write(f'#### La predicción es: {"Crítica" if prediction[0] else "No Crítica"}')
    with st.spinner("Calculando recomendaciones..."):

        recommendations, vote_counts, = recommend_changes(input_df, model)

        percentage = "%.1f" % ((vote_counts[1] / (vote_counts[1] + vote_counts[0])) * 100)
        st.write(f"Esta vacante fue clasificada como crítica por {percentage}% de los {vote_counts[0] + vote_counts[1]} modelos")

        dummies_time_of_day = [i for i in input_df.columns if "time_of_day" in i]
        dummies_education_level = [i for i in input_df.columns if "education_level" in i]
        dummies_exp = [i for i in input_df.columns if "vacant_experience" in i]

        listas=dummies_time_of_day + dummies_education_level + dummies_exp
        recommendations=pd.DataFrame(recommendations).T
        recommendations=recommendations.groupby(["feature_name", "feature_value", "direction"]).agg({"threshold_value":"mean", "gini_change":"mean"}).reset_index()
        recommendations=recommendations[~recommendations["feature_name"].isin(listas)]
        if prediction[0]:
            st.write("Para mejorar esta vacante, considere ajustar las siguientes características:")
            recommendations=recommendations.sort_values("gini_change", ascending=False).head(3)
            #where feature_value = False then threshold_value=True
            recommendations["threshold_value"]=np.where(recommendations["feature_value"]==False, True, recommendations["threshold_value"])
            recommendations["threshold_value"]=np.where(recommendations["feature_value"]==True, False, recommendations["threshold_value"])

            st.write(recommendations)
        st.success("✅ Done!")
            # Display the most important recommendations


            # for feature_name, details in recommendations:
            #     if feature_name not in (dummies_time_of_day + dummies_education_level + dummies_exp):
            #         st.write(f"- {feature_name}: Consider to {details['direction']} from {details['feature_value']} to {'>' if details['direction'] == 'increase' else '<'} {details['threshold_value']} (Cambio en Gini: {details['gini_change']:.2f}, Importancia: {details['importance']:.2f})")
