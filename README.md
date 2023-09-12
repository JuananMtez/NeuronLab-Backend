<!-- PROJECT LOGO -->
<div align="center">

  <h3 align="center">NeuronLab Framework (Backend)</h3>
  <p align="center">
    Brain Computer Interface (BCI) Framework for the study of biosignals.
  </p>
</div>


## About the project


NeuronLab is a secure, multiplatform, and web-based framework that provides a working environment to researchers where every phase of the BCI lifecycle (acquisition, preprocessing, features extraction, and classification) can be performed. Additionally, it defines sophisticated tools to study the EEG signals.

### Built With
![python] ![FastAPI]

## Getting Started

### Prerequisites
* Python 3.8.
* MySQL 


### Installation
1. Clone the repo.
```sh
git clone https://github.com/JuananMtez/NeuronLab-Backend.git
```

2. Change to project directory.
```sh
cd NeuronLab-Backend
```

4. Install dependencies.
```shell
pip install -r requirements.txt
```

5. Create database in MySQL.

6. Modify properties in ```./app/config/properties.ini``` .
```ini
[DATABASE]
user = TO BE DEFINED
password = TO BE DEFINED
host = TO BE DEFINED
database = TO BE DEFINED

[SECURITY]
secret_key = TO BE DEFINED
algorithm = TO BE DEFINED
access_token_expire_minute = TO BE DEFINED
```

| parameter                    |   Description   |
|:-----------------------------|:---------------:|
| user                         | 	MySQL user 
| password	                    |  	MySQL password 
| host 	                       |       	MySQL host       
| database 	                   |       	MySQL database         
| secret_key 	                 |       	Key used to sign the JWT tokens       
| algorithm 	                  |      	Algorithm used to sign the JWT token   
| access_token_expire_minute 	 |       	Token lifetime in minutes
 	        



## Usage

Run NeuronLab-Backend.
```shell
python3 -m uvicorn app.main:app
```

The tables will be created in the database automatically.





## Author

* **Juan Antonio Martínez López** - [Website](https://juananmtez.github.io/) - [LinkedIn](https://www.linkedin.com/in/juanantonio-martinez/)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

[Python]: https://img.shields.io/badge/Python-20232A?style=for-the-badge&logo=python
[FastAPI]: https://img.shields.io/badge/fastapi-20232A?style=for-the-badge&logo=fastapi