from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import os
import pickle

# Load the dataset
Students_data = pd.read_csv('Students_data.csv')

# Convert relevant columns to numeric
numeric_columns = ['Xth_Grade_Score', 'XIIth_Grade_/_Diploma_Score', 'Mht-Cet_Percentile']
for column in numeric_columns:
    Students_data[column] = pd.to_numeric(Students_data[column], errors='coerce')

# Drop rows with NaN values after conversion
Students_data = Students_data.dropna(subset=numeric_columns)
# Generate the unique names and branches for dropdown options
unique_names = Students_data['Name'].unique().tolist()
unique_branches = Students_data['Branch'].unique().tolist()
# Load the trained RandomForestClassifier model from .pkl file
with open('trained_model.pkl', 'rb') as file:
    clf = pickle.load(file)

# Generate the plots
def generate_plots():
    plot_divs = []

    # Distribution of Mht-Cet Percentile
    fig1 = px.histogram(Students_data, x='Mht-Cet_Percentile',
                        title='Distribution of Mht-Cet Percentile',
                        labels={'Mht-Cet_Percentile': 'Mht-Cet Percentile'},
                        width=735)

    div1 = go.Figure(fig1).to_html(full_html=False)
    plot_divs.append(div1)

    # Comparison of Xth and XIIth Grade Scores
    fig2 = px.scatter(Students_data, x='Xth_Grade_Score', y='XIIth_Grade_/_Diploma_Score', color='Branch',
                      title='Comparison of Xth and XIIth Grade Scores',
                      labels={'Xth_Grade_Score': 'Xth Grade Score', 'XIIth_Grade_/_Diploma_Score': 'XIIth Grade Score'},
                      width=735)

    div2 = go.Figure(fig2).to_html(full_html=False)
    plot_divs.append(div2)

    # Top 10 Students by Mht-Cet Percentile
    top_10_students = Students_data.nlargest(10, 'XIIth_Grade_/_Diploma_Score')
    fig3 = px.bar(top_10_students, x='Name', y='XIIth_Grade_/_Diploma_Score',
                  title='Top 10 Students by XIIth_Grade_/_Diploma_Score',
                  labels={'XIIth_Grade_/_Diploma_Score': 'XIIth_Grade_/_Diploma_Score', 'Name': 'Student Name'})

    div3 = go.Figure(fig3).to_html(full_html=False)
    plot_divs.append(div3)

    # List of graph types from the provided function
    graph_types = ['histogram', 'violin', 'countplot', 'pairplot', 'scatter']

    for graph_type in graph_types:
        if graph_type == 'histogram':
            fig = px.histogram(Students_data, x='Xth_Grade_Score', title='Xth Grade Score Distribution', nbins=20)
        elif graph_type == 'violin':
            fig = px.violin(Students_data, y='Xth_Grade_Score', title='Xth Grade Score Distribution')
        elif graph_type == 'countplot':
            fig = px.histogram(Students_data, x='Branch', title='Number of Students by Branch')
        elif graph_type == 'pairplot':
            fig = px.scatter_matrix(Students_data, dimensions=['Xth_Grade_Score', 'XIIth_Grade_/_Diploma_Score'], color='Branch')
        elif graph_type == 'scatter':
            fig = px.scatter(Students_data, x='Xth_Grade_Score', y='XIIth_Grade_/_Diploma_Score', title='Xth vs XIIth Grade Score')

        plot_div = fig.to_html(full_html=False)
        plot_divs.append(plot_div)

    return plot_divs

app = Flask(__name__)

@app.route('/')
def index():
    plot_divs = generate_plots()
    return render_template('index.html', plot_divs=plot_divs, names=unique_names, branches=unique_branches)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        mht_cet_percentile = float(request.form['mht_cet_percentile'])
        xii_marks = float(request.form['xii_marks'])

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'Xth_Grade_Score': [0],
            'XIIth_Grade_/_Diploma_Score': [xii_marks],
            'Mht-Cet_Percentile': [mht_cet_percentile]
        })

        # Predict branch using the trained model
        y_pred = clf.predict(input_data)

        # Decode numerical label to original branch name based on provided mapping
        branch_mapping = {
            0: 'CE',
            1: 'DS',
            2: 'AIML',
            3: 'EXTC',
            4: 'MCA',
            5: 'No Branch'
        }
        predicted_branch = branch_mapping.get(y_pred[0], 'Unknown Branch')

        return jsonify({'predicted_branch': predicted_branch})
    
    return render_template('predict.html')


@app.route('/student_details', methods=['GET', 'POST'])
def student_details():
    selected_branch = request.form.get('branch', 'CE')
    selected_name = request.form.get('name', '')

    student_details = Students_data[(Students_data['Branch'] == selected_branch) & (Students_data['Name'] == selected_name)].to_dict('records')
    
    if student_details:
        student_details = student_details[0]
        return render_template('student_details.html', student_details=student_details, names=unique_names, branches=unique_branches)
    else:
        return render_template('student_details.html', error="No details found for the provided student and branch.", names=unique_names, branches=unique_branches)


if __name__ == '__main__':
    app.run(debug=True)
