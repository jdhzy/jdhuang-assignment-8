# Assignment 8: Logistic Regression

---
# Youtube link: https://youtu.be/hopoxz3EBdY

**Goals**:

In this assignment, you'll explore the effect of shifting clusters in a dataset on the parameters of a logistic regression model. You will implement parts of the code to:
1. Generate datasets with shifted clusters.
2. Fit a logistic regression model and extract parameters.
3. Visualize the data, decision boundary, and logistic regression results.
4. Analyze how these parameters change with increasing shift distances.

## Part 0: Setup Environment

You can use the `Makefile` to install all dependencies. In your terminal, simply run:

```bash
make install
```

This will automatically install the necessary packages listed in `requirements.txt`, including:

- flask
- numpy
- scikit-learn
- scipy
- matplotlib

## Part 1: Implementing Logistic Regression

1. **Generate Clusters with a Shift**  
- Implement the code to shift the second cluster along both the x-axis and y-axis by a specified `distance` parameter. This step will simulate different levels of separation between clusters, which you will explore later in the assignment.

2. **Record Parameters for Each Shift Distance**  
- Fit a logistic regression model to each generated dataset, and then extract and record the intercept (`beta0`) and coefficients (`beta1`, `beta2`) and any other necessary informtaion for each shift distance.

3. **Plot Each Dataset and Decision Boundary**  
- Implement code to plot the data points for each class in different colors. Include the decision boundary calculated from `beta0`, `beta1`, and `beta2` values to visually separate the classes.

4. **Calculate Logistic Loss for Each Model**  
- Implement code to compute the logistic loss for each shift distance. This loss reflects the accuracy of the logistic regression model at classifying the points in each dataset.

5. **Plot Results Across Shift Distances**  
- Implement code to create multiple plots that show how model parameters (`beta0`, `beta1`, `beta2`), slope, intercept, logistic loss, and margin width change as the shift distance increases.    

## Part 2: Testing Your Code with a Static Input (Optional)

1. If you prefer, you can also test the code locally by running the script directly and specifying necessary parameters. 

2. Run the script in your terminal:
   
   python logistic_regression.py

3. Check the output image in the `results` folder.

## Part 3: Running the Interactive Module

Once the environment is set up, you can start the Flask application by running:

```bash
make run
```

This will start the Flask server and make the interactive application available locally at `http://127.0.0.1:3000`.

1. Open your browser and go to `http://127.0.0.1:3000`.
2. Choose the range of the shift distance by specifying the lower bound (inclusive), the upper bound (inclusive), and the total step number of shifts.
3. The resulting figure will be displayed.

---
***Submission***

1. Create a Short Demo Video (1-2 minutes):

- Create a demo video by screen recording your output with your voice-over.
- Explain any patterns you observe and discuss what happens when the shift increases.
- Explain the relation between the plotted parameters.

2. Submit both your completed code and the demo video link.
You can either embed the demo video in your portfolio website or just create an unlisted YouTube video with a link to that YT video in your assignment 8 github repo's readme.

