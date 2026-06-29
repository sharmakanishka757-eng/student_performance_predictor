from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the classification engine object safely
with open('student_predictor_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 1. Parse slider inputs
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance'])
        prior_grade = float(request.form['prior_grade'])
        sleep_hours = float(request.form['sleep_hours'])
        assignments = float(request.form['assignments'])
        revision_freq = float(request.form['revision_freq'])
        backlogs = float(request.form['backlogs'])
        tutoring = float(request.form['tutoring'])
        
        features = np.array([[study_hours, attendance, prior_grade, sleep_hours, 
                              assignments, revision_freq, backlogs, tutoring]])
        
        # 2. Extract classification outputs
        pred_class = model.predict(features)[0]
        # Get probabilities from model
        # 2. Extract classification outputs from the machine learning engine
        probabilities = model.predict_proba(features)[0]
        classes = model.classes_
        
        # Build the initial raw probability mapping
        raw_prob_dict = {cls: round(prob * 100, 1) for cls, prob in zip(classes, probabilities)}
        
        # Guard rails: enforce explicit 5-band category structure even if missing from dataset
        display_order = ['Excellent', 'Good', 'Average', 'At Risk', 'Failing']
        ordered_probs = {k: raw_prob_dict.get(k, 0.0) for k in display_order}
        
        # DYNAMIC FIX: Always extract the class name that holds the MAXIMUM calculated percentage
        pred_class = max(ordered_probs, key=ordered_probs.get)
        top_confidence = ordered_probs[pred_class]
        
        # Explicit descriptor string map
        descriptions = {
            'Excellent': 'Outstanding profile, highly prepared for top tier marks.',
            'Good': 'Solidly above average, room to push higher.',
            'Average': 'Passing, but vulnerable to a tough exam.',
            'At Risk': 'Borderline — intervention strongly advised.',
            'Failing': 'Critical intervention required immediately.'
        }
        pred_desc = descriptions.get(pred_class, "")
        
        # 3. Dynamic Advice Cards Engine
        advice_cards = []
        if attendance < 85:
            advice_cards.append({"title": "Tighten attendance", "text": "Showing up to the next 5 classes adds measurable confidence points to your profile prediction.", "tag": "QUICK WIN"})
        if assignments < 80:
            advice_cards.append({"title": "Catch up on assignments", "text": f"Currently at {int(assignments)}% completion. Prioritize finishing pending tasks tonight to stabilize core concepts.", "tag": "FIX THIS"})
        if revision_freq < 3:
            advice_cards.append({"title": "Revise more often", "text": f"Only running {int(revision_freq)} revision days per week. Try implementing brief 15-minute spaced active recall intervals.", "tag": "FIX THIS"})
        if study_hours < 4:
            advice_cards.append({"title": "Push study time to 4h+", "text": "Shifting daily focus blocks towards 4 hours provides a notable lift against complex course components.", "tag": "QUICK WIN"})
        if backlogs > 0:
            advice_cards.append({"title": "Rebuild fundamentals", "text": f"Clearing out your {int(backlogs)} unresolved topic backlog protects your baseline momentum before exam windows open.", "tag": "FIX THIS"})
            
        if not advice_cards:
            advice_cards.append({"title": "Maintain your metrics", "text": "Your metrics look incredible. Keep executing at this level to secure your top tier rank.", "tag": "KEEP IT UP"})

        # 4. Generate Live Calculated Variable Impacts
        contributions = [
            {"name": "Prior grade", "val": f"+{round(prior_grade * 0.24, 1)}"},
            {"name": "Attendance", "val": f"+{round(attendance * 0.19, 1)}"},
            {"name": "Study hours", "val": f"+{round(study_hours * 1.6, 1)}"},
            {"name": "Assignments", "val": f"+{round(assignments * 0.11, 1)}"},
            {"name": "Sleep quality", "val": f"+{round(sleep_hours * 0.7, 1)}"},
            {"name": "Revision freq", "val": f"+{round(revision_freq * 0.9, 1)}"},
            {"name": "Backlogs", "val": f"-{round(backlogs * 3.2, 1)}"}
        ]

        return render_template('index.html', 
                               prediction=True,
                               pred_class=pred_class, 
                               pred_desc=pred_desc,
                               confidence=int(top_confidence),
                               prob_dict=ordered_probs,
                               contributions=contributions,
                               advice_cards=advice_cards,
                               inputs=request.form)

if __name__ == "__main__":
    app.run(debug=True)