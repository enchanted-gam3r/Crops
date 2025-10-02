import gradio as gr
import pandas as pd
import joblib

#Loading Model and Encoder
model_pipeline = joblib.load('crop_suggestion_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')
  
def predict_crop(soil_ph, nitrogen, phosphorus, potassium, organic_carbon,rainfall, temperature, humidity, irrigation, season, landholding):
    input_data = {
        'Soil_pH': soil_ph,
        'Nitrogen_N': nitrogen,
        'Phosphorus_P': phosphorus,
        'Potassium_K': potassium,
        'Organic_Carbon': organic_carbon,
        'Rainfall_mm': rainfall,
        'Temperature_C': temperature,
        'Humidity_%': humidity,
        'Irrigation': irrigation,
        'Season': season,
        'Landholding_ha': landholding
    }

    input_df = pd.DataFrame([input_data])
    prediction_encoded = model_pipeline.predict(input_df)
    predicted_crop = label_encoder.inverse_transform(prediction_encoded)

    return f"The suggested crop is: {predicted_crop[0]}"

#Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌾 Jharkhand Crop Suggestion Model 🌳
        Enter the environmental and soil conditions to receive a recommendation for the best crop to plant.
        This model is trained on data specific to Jharkhand.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Soil & Nutrient Conditions")
            soil_ph = gr.Slider(minimum=4.0, maximum=9.0, value=6.5, label="Soil pH", info="Enter the soil pH value (4-9)")
            nitrogen = gr.Slider(minimum=20, maximum=180, value=90, label="Nitrogen (N) Content (kg/ha)")
            phosphorus = gr.Slider(minimum=5, maximum=80, value=42, label="Phosphorus (P) Content (kg/ha)")
            potassium = gr.Slider(minimum=10, maximum=210, value=43, label="Potassium (K) Content (kg/ha)")
            organic_carbon = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=0.8, label="Organic Carbon (%)")

        with gr.Column():
            gr.Markdown("### Environmental & Farming Conditions")
            rainfall = gr.Slider(minimum=200, maximum=2500, value=1200, label="Annual Rainfall (mm)")
            temperature = gr.Slider(minimum=10.0, maximum=45.0, value=27.5, label="Average Temperature (°C)")
            humidity = gr.Slider(minimum=20, maximum=100, value=80, label="Average Humidity (%)")
            irrigation = gr.Dropdown(choices=['Rainfed', 'Canal', 'TubeWell', 'Pond'], value='Rainfed', label="Irrigation Method")
            season = gr.Dropdown(choices=['Kharif', 'Rabi', 'Zaid'], value='Kharif', label="Season")
            landholding = gr.Slider(minimum=0.1, maximum=10.0, step=0.1, value=2.5, label="Land Holding Size (ha)")

    with gr.Row():
        predict_btn = gr.Button("Suggest Crop", variant="primary")

    with gr.Row():
        output_text = gr.Textbox(label="Model Recommendation", interactive=False)

    predict_btn.click(
        fn=predict_crop,
        inputs=[
            soil_ph, nitrogen, phosphorus, potassium, organic_carbon,
            rainfall, temperature, humidity, irrigation, season, landholding
        ],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()
