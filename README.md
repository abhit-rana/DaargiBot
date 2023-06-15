## Introduction

A Medical Disease Diagnosis Chatbot where the user first self reports the symptoms and then the Chatbots inquire more symptoms to arrive at the diagnosis result.

Currently the chatbot supports following 12 Diseases Medical Diagnosis:

Asthma,
Coronary heart disease, 
Pneumonia, 
Traumatic brain injury, 
Esophagitis, 
Enteritis, 
Rhinitis, 
Thyroiditis, 
Dermatitis, 
External otitis, 
Conjunctivitis, and 
Mastitis

_Note_: _We implemented our Chatbot in [MarkovML](https://www.markovml.com/) software, so this repo would also contain code of using the features like Experiment Tracking, and Model Recording._

## Architecture of DaargiBot:

Every Chatbot has three components: NLU, DM and NLG

We implemented NLU through RASA Framework,

DM through Rule Based Technique,

and NLG through Template Based Method

Here is the Architecture Describing things Inside each component and Flow Diagram of DaargiBot:

![DaargiBot](https://user-images.githubusercontent.com/88608893/221353051-6c5ebdae-86f4-4d19-bf13-dbe89b70084c.png)

## Contents of Repo

* [Data](https://github.com/abhit2code/DaargiBot/tree/main/data): This Folder contains the [Data](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) used for our Bot. It consist of csv files for Disease Description, Symptoms of a particular Disease, Severity Scores of each symptom, and Disease Precaution.
* [NLU](https://github.com/abhit2code/DaargiBot/tree/main/NLU): contains file for understanding user message. The code here is used for extracting the user intent and entities related to that particular intent. We made the NLU component through [RASA](https://rasa.com/docs/rasa/) framework.
* [DM](https://github.com/abhit2code/DaargiBot/tree/main/DM): This folder contains file for deciding what action i.e. whether to predict the Disease or inquire more about the symptoms etc should be taken by bot at certain state of the Chatbot Flow.
* [NLG](https://github.com/abhit2code/DaargiBot/tree/main/NLG): Contains the file to generate a Natural Language Response for the output from DM
* [Interaction](https://github.com/abhit2code/DaargiBot/tree/main/Interaction): This is the file which combines all the three components of the Bot so the user could interact with it.

## Installing the Chatbot

**1. Downloading the Repo**

* Go to the Homepage of this Repo. Download the ZIP FILE from the "<> Code" button there.
  
  ![tempsnip](https://user-images.githubusercontent.com/123395972/227699641-fcd9f9ea-4be0-4737-aba7-fb2524205b05.jpg)
  
 * Extract the ZIP file where you want
 * Rename the Folder according to your wish. **Lets rename the Downloaded folder(DaargiBot-main) to be - DaargiBot**

**2. Setting up virtual environment**

```
pip3 install virtualenv
virtualenv DaargiBot
```

**3. Opening up the Repo**
```
cd DaargiBot
```

**4. Activating Virtual Environment**
```
source bin/activate
```

**5. Downloading the Required Python Packages**
```
pip3 install -r requirements.txt
```

Hurray we are all set up!!

## Running the Chatbot

Start your terminal and say the current terminal tab(in your mind) to be t1

**1. Open up the DaargiBot Repo**
```
cd DaargiBot
```

**2. Activating Virtual Environment**
 ```
 source bin/activate
 ```
 
 **3. Run the run1.sh file**
 ```
 bash run1.sh
 ```
  
**4. Open up another terminal tab and say it to be "t2"**

_Further Command would be runned on this new terminal tab->t2_

**5. Repeat Steps 1, and 2**

**6. Run the run2.sh file**
```
bash run2.sh
```

Hurray! we can Now Interact with the Chatbot but do read the Following-> "Instructions to Use" section

## Instructions to Use the Chatbot

Before Reading this do have look at the Architecture Section of the this Chatbot

There are six stages in our Chatbot
![DaargiBot - Page 2 (1)](https://user-images.githubusercontent.com/123395972/230561599-402b4e4d-bfb9-4648-9920-f1b533dbf74e.jpeg)

* S1: Here the user may enter the Greeting Message. 
  
  example: Hello or Hi. the chatbot then replies with a return greeting message.
  
  ![image](https://user-images.githubusercontent.com/123395972/227702694-a56348bf-9568-46b3-b04f-b59b7d108198.png) 
 
* S2: The user needs to enter atleast one symptom which are being observed to them or some other person.
  
  example: I am(or my child, mom, father etc) experiencing(or suffering, facing, noticing etc) cough in the morning(or any other symptom) and fatigue in the 
  evening(or any other symptom). What could be the issue?
  
  ![image](https://user-images.githubusercontent.com/123395972/227703697-787bf457-6e33-405b-96c3-b0f4b6679967.png)

* S3: This stage follows the self-report or S2. If the symptoms reported in stage are not strong enough to predict the diesease then the chatbot in return ask 
  whether the patient suffered a particular symptom. This process continues until the chatbot gather enough information about the symptoms to predict the 
  disease. **Here the user needs to enter whether the patient suffered the symptom asked by the Chatbot**
  
  example: 
  
  _for yes_: yes, often, sometimes, etc.
  _for no_: no, don't, not, not very often, etc.
  
  ![image](https://user-images.githubusercontent.com/123395972/227703967-3f325135-87f0-4f18-8da3-07becc982325.png)
  
* S4: The chatbot predicts the Possible Disease after gaining enough information about symptoms.
  
  ![image](https://user-images.githubusercontent.com/123395972/227704810-94da61e6-d7d4-42cf-8277-35f2d9be4b74.png)
  
  The chatbot then provide suggestion by asking for the number of days symptoms are observed. User needs to enter the number of days
  
  ![image](https://user-images.githubusercontent.com/123395972/227704819-6910eceb-356e-48a7-96b9-8fa10aa58c1c.png)
  
  Through Number of days of suffering chatbot calculates the severity score.
  
  If the condition is less severe then suggest precautions
  
  ![image](https://user-images.githubusercontent.com/123395972/227704839-312ef092-ecd9-4581-9262-d93d3e537cef.png)
  
  If the condition is severe then ask for consulation
  
  ![image](https://user-images.githubusercontent.com/123395972/227704891-14010478-8486-40cf-a08c-6f3c11f026a0.png)

* S5: The user shows the gratitude to the chatbot and the chatbot in turn provides supporting message.

  example: thanks, tq for the help, etc

  ![image](https://user-images.githubusercontent.com/123395972/227710135-89f4afa2-e246-4a6b-81ac-5e0dccbc2ceb.png)
  
* S6: Here the user enters the leaving message and the chatbot also replies with the leaving message

  example: Bye
  
  ![image](https://user-images.githubusercontent.com/123395972/227710148-e0bb23b9-2996-4c14-b05e-b36ad866f34d.png)




SWOT ANALYSIS

![SWOTANALYSIS](https://user-images.githubusercontent.com/88608893/221353122-c75ed90a-d50f-4047-95df-9cef878e6576.JPG)


## Limitation

While Daargi bot has the potential to enhance healthcare accessibility and ease the burden on healthcare providers, it has certain limitations that need to be considered. One of the major constraints is its limited accuracy due to its reliance on algorithms and natural language processing. The accuracy of its diagnosis is restricted by the quality and quantity of data available, and it can only predict the 41 diseases for which it has been trained. Therefore, there is a possibility that Daargi bot may provide inaccurate information in some cases.

Another limitation of Daargi bot is its limited training on 132 symptoms, which may not be sufficient to capture all possible variations in symptoms. An unusual symptom that is not included in the training dataset might lead to skewed results. Moreover, Daargi bot is limited to predicting only a small set of diseases, and patients with complex or multiple health conditions may require a more comprehensive evaluation by a healthcare provider.

## Future Scope
We plan on improving upon this idea as we see the potential it has to revolutionise the medical marketand even the way we understand and tackle healthcare as a whole, despite some glaring issues it has right now. Our first step would be to expand our database, add new symptoms and map them to new diseases currently out of the scope of the DaargiBot. For this we plan to acquire medical records and databases. The other major step we plan on taking at the earliest is to improve upon our current data storage protocols and integrate better privacy and security measures to uphold the privacy and safeguard the sensitive data of our consumers. We would need to get in touch with a team of medical professionals and practicing lawyers to tackle the legal issues and the data acquisition problem. With our team fully assembled and above mentioned concerns dealt with, the DaargiBot shall not be limited to only being a diagnosis system but can also be upgraded to recommend medications and test procedures, thus reducing the workload of medical professionals, cutting down the cost incurred by the patients and streamlining the entire process.

## Instructions to Contribution

To contribute towards the improvement of Daargi Bot, please follow the steps below:
* Fork the repository: This creates a copy of the original repository on your GitHub account that you can work on.
* Clone the repository: Clone the repository to your local machine by running
git clone <repository-url>
in your terminal. This will create a local copy of the repository that you can work on.
* Create a new branch: Before making any changes, create a new branch by running
git checkout -b <branch-name>
in your terminal. This will create a new branch with the name <branch-name> that you can work on. Always make changes in a new branch rather than the main branch.
* Make changes: Make changes to the code in your local copy of the repository. Once you are done making changes, save and commit them using the command 
git add.
to add all changed files and 
git commit -m "your commit message" 
to commit your changes with a descriptive message.
* Push changes: Once you have committed your changes locally, push them to your forked repository by running 
git push origin <branch-name> 
in your terminal.
* Open a pull request: Once you have pushed your changes to your forked repository, open a pull request on the original repository. The owner of the original repository will then review your changes and either merge them or request changes.
* Collaborate: If the owner of the repository requests changes, make the necessary changes in your local copy of the repository and push them to your forked repository. Once you are satisfied with your changes, open a new pull request.
Remember to always communicate with your collaborators and follow any specific guidelines or conventions set by the repository owner.

## Contributors
* [Ieesha Deshmukh](https://github.com/ieesha1102)
* [Amar Gwari](https://github.com/AmarGwari)
