# Data Acquisition and Processing Tools

This repository contains three Jupyter Notebook files designed to interact with an API for retrieving and processing data. The workflow includes obtaining an authentication token, fetching data from the API, and converting the API response into a CSV file for further analysis.

## File Description

- **gettoken.ipynb**  
  This notebook is responsible for obtaining an authentication token from the API. It contains code to send a request to the authentication endpoint. Running this notebook is a prerequisite for executing the other notebooks, as it provides the valid token required for subsequent API requests.

- **getData.ipynb**  
  This notebook utilizes the token acquired from `gettoken.ipynb` to fetch data from the API. It demonstrates how to configure API request parameters such as the URL, HTTP method, headers, and any required parameters. Adjust these configurations as needed to suit your API requirements.

- **GenerateCSVfromResp.ipynb**  
  This notebook processes the API response obtained via `getData.ipynb` and converts the data into CSV format. It leverages Python libraries (e.g., pandas) to clean and transform the data, making it ready for analysis, visualization, or storage.

## Requirements

- Python 3.x
- Jupyter Notebook or Jupyter Lab

### Dependencies

Ensure the following Python libraries are installed. If not, you can install them using pip:

```bash
pip install requests pandas
