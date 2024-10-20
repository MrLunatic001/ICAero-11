from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/cost', methods=['POST'])
def calculate_cost():
    data = request.json
    lat = data.get('lat')
    lng = data.get('lng')

    # Call your CostFunction.py script with lat and lng as arguments
    # Assuming CostFunction.py returns a single cost value
    try:
        # Replace 'python' with 'python3' if needed
        result = subprocess.run(['python', 'CostFunction.py', str(lat), str(lng)], capture_output=True, text=True)
        cost_value = float(result.stdout.strip())  # Parse the cost value from output
        return jsonify({'cost': cost_value})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)