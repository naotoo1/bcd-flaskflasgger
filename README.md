# bcd-Flask and Flasgger
An end-to-end implementation of Breast Cancer Detection using [prosemble ML package](https://github.com/naotoo1/prosemble) within the Flask framework and Flasgger

## How to use
To diagnose breast cancer disease and return the confidence of the diagnosis,
1. Run the app.py  and get the local host ``` http://local host/apidocs ```
2. To  predict for a single test case, click on ```Get``` ---> ```Try it out ```
3. Enter the values for Radius_mean, Radius_texture and Method either as soft or hard
4. click on ```execute ``` to get diagnosis with confidence

To diagnose from multiple inputs from a file
1. Click on ```POST ``` ---> ```Try it out ```
2. Select your input file from its location and enter the Method either as soft or hard
3. Click on ```execute ``` to get diagnosis with confidence for each input in the file.

### FastAPI framework Version
For fastapi framework deployment version refer to [bcd-fastapi](https://github.com/naotoo1/bcd-fastapi)

### FlaskPyWebIO framework Version
For FlaskPyWebIO framewokr version refer to [bcd-FlaskPyWebIO](https://github.com/naotoo1/bcd-FlaskPyWebIO)


