from pyexpat import model
import streamlit as st
import pickle 
import numpy as np
with open('model.pkl', 'rb') as f:
    model =pickle.load(f) 
def cancer_prediction(mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_concavity,mean_concave_points,mean_symmetry,mean_fractal_dimension,radius_error,texture_error,perimeter_error,area_error,smoothness_error,compactness_error,concavity_error,concave_points_error,symmetry_error,fractal_dimension_error,worst_radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension):
    input_data = [mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_concavity,mean_concave_points,mean_symmetry,mean_fractal_dimension,radius_error,texture_error,perimeter_error,area_error,smoothness_error,compactness_error,concavity_error,concave_points_error,symmetry_error,fractal_dimension_error,worst_radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension]
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    
   
    return (prediction)


def main():
    st.title('The Test')
    
   
    
    mean_radius=st.text_input('Mean radius','type here')
    mean_texture=st.text_input('mean_texture','type here')
    mean_perimeter=st.text_input('mean_perimeter','type here')
    mean_area=st.text_input('mean_area','type here')
    mean_smoothness=st.text_input('mean_smoothness','type here')
    mean_compactness=st.text_input('mean_compactness','type here')
    mean_concavity=st.text_input('mean_concavity','type here')
    mean_concave_points=st.text_input('mean_concave_points','type here')
    mean_symmetry=st.text_input('mean_symmetry','type here')
    mean_fractal_dimension=st.text_input('mean_fractal_dimension','type here')
    radius_error=st.text_input('radius_error','type here')
    texture_error=st.text_input('texture_error','type here')
    perimeter_error=st.text_input('perimeter_error','type here')
    area_error=st.text_input('area_error','type here')
    smoothness_error=st.text_input('smoothness_error','type here')
    compactness_error=st.text_input('compactness_error','type here')
    concavity_error=st.text_input('concavity_error','type here')
    concave_points_error=st.text_input('concave_points_error','type here')
    symmetry_error=st.text_input('symmetry_error','type here')
    fractal_dimension_error=st.text_input('fractal_dimension_error','type here')
    worst_radius_worst=st.text_input('worst_radius_worst','type here')
    texture_worst=st.text_input('texture_worst','type here')
    perimeter_worst=st.text_input('perimeter_worst','type here')
    area_worst=st.text_input('area_worst','type here')
    smoothness_worst=st.text_input('smoothness_worst','type here')
    compactness_worst=st.text_input('compactness_worst','type here')
    concavity_worst=st.text_input('concavity_worst','type here')
    concave_points_worst=st.text_input('concave_points_worst','type here')
    symmetry_worst=st.text_input('symmetry_worst','type here')
    fractal_dimension=st.text_input('fractal_dimension','type here')
    
  
    if st.button("Predict"):
        output=cancer_prediction(mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_concavity,mean_concave_points,mean_symmetry,mean_fractal_dimension,radius_error,texture_error,perimeter_error,area_error,smoothness_error,compactness_error,concavity_error,concave_points_error,symmetry_error,fractal_dimension_error,worst_radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension)
        st.success(output)
        

if __name__=='__main__':
      main()



   


