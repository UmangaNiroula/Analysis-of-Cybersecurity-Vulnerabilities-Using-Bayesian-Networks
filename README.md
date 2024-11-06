# Elevator Pitch:

As cyber threats become increasingly sophisticated, understanding vulnerabilities is essential for defending against attacks. This project analyzes a 2022 dataset of cybersecurity vulnerabilities to uncover patterns in products, vendors, and attack vectors that are most frequently targeted. By using Bayesian networks and classification analysis, we reveal relationships between factors like vulnerability severity, product category, and response time. These insights are designed to help organizations prioritize security efforts, improve patch management, and bolster their cybersecurity defenses against emerging threats, ultimately guiding more resilient risk management strategies.

# Cyber Threat Analysis Using Bayesian Networks

1. Introduction
Cybersecurity continues to evolve as organizations face an increasing number of sophisticated threats. As technology advances, so do the tactics of malicious actors who exploit vulnerabilities in software and systems (Samtani et al., 2020). In 2022, vulnerabilities reported to the Cybersecurity and Infrastructure Security Agency (CISA) revealed critical weaknesses in various products, sparking significant concern (Rothman and Rothman, 2020). Understanding these vulnerabilities is essential for developing defense strategies, mitigating risks, and improving response times. In this report, we analyze a comprehensive dataset of cybersecurity vulnerabilities from 2022. Using Bayesian networks and extensive classification analysis, we aim to uncover relationships between key factors such as severity, product, vendor, and attack vector. Our goal is to generate insights that can help prioritize responses to vulnerabilities and strengthen the overall cybersecurity posture of organizations.
To gain deeper insights, we first examine the most frequently targeted products and vendors, identifying industries most affected by these vulnerabilities. By analyzing trends across sectors, we can uncover patterns that may signal emerging threats. We also explore the methods attackers use—such as phishing, social engineering, and zero-day exploits—to understand the most prevalent attack vectors. A key focus is the severity of vulnerabilities and how critical vulnerabilities are distributed across product categories. Bayesian networks allow us to model probabilistic relationships between these factors, offering a clearer understanding of how different elements interact in cybersecurity. Additionally, we assess organizational response times and evaluate mitigation strategies like patch management, aiming to identify best practices that reduce risk. Ultimately, our findings will inform risk management frameworks, helping organizations strengthen their defenses against evolving threats.


2. Objective

The primary objectives of this study are:
To analyze cybersecurity vulnerabilities from 2022, focusing on products and vendors most affected.
To explore relationships between vulnerability features such as severity, vector, complexity, and due dates.
To use Bayesian networks to model and understand the conditional dependencies between various features of vulnerabilities.
To perform classification analysis using a Naive Bayes classifier to predict the severity of a vulnerability based on its features.
To derive actionable insights regarding the most vulnerable products and efficient patching strategies. 


3. Dataset Overview

The dataset used in this study is sourced from the Cybersecurity and Infrastructure Security Agency (CISA) (Petkova, 2021). It contains detailed information on vulnerabilities reported in 2022, including data on:

    CVE ID: A unique identifier for each vulnerability.
    
    Vendor/Project: The vendor or project associated with the vulnerability.
    
    Product: The specific product affected by the vulnerability.
    
    Vulnerability Name: A brief description or name of the vulnerability.
    
    Date Added: The date the vulnerability was added to the database.
    
    Severity: The severity level assigned to the vulnerability (high, medium, low).
    
    CVSS: The Common Vulnerability Scoring System score, quantifying the severity of the vulnerability.
    
    Vector: The attack vector used to exploit the vulnerability.
    
    Complexity: The complexity level of exploiting the vulnerability (high, low).
    
    Required Action: Recommended actions to mitigate or resolve the vulnerability.

Dataset Details:
Link: https://www.kaggle.com/datasets/thedevastator/exploring-cybersecurity-risk-via-2022-cisa-vulne
Data Source: Kaggle 
Rows: 398,416
Columns: 16

The dataset provides a robust foundation for statistical analysis, correlation exploration, and machine learning modeling.

4. Methodology

4.1 Data Preprocessing and Cleaning
Before conducting any analysis, we performed the following data preparation steps:
Handling Missing Data: Missing values in critical columns like severity, CVSS score, and vector were imputed using statistical methods (median imputation for numeric values and mode imputation for categorical variables). This ensured that the dataset remained robust and consistent, allowing for meaningful analysis without introducing bias due to missing information (Mallinckrodt, 2013).

Data Transformation: Date variables (e.g., date_added, due_date, pub_date) were converted to a standard format to maintain uniformity across all records and simplify time-based calculations. Categorical variables such as severity, vector, and complexity were also encoded using techniques like one-hot encoding or label encoding, depending on the analysis requirements, to make them suitable for machine learning models Mallinckrodt, 2013).

Feature Engineering: New features were derived, such as patching_time (calculated as the difference between date_added and due_date), to capture insights into the timeliness of addressing vulnerabilities and the efficiency of patch management. Additional features like vulnerability_age (time since discovery) were also created to provide further insights into the potential risk exposure period Mallinckrodt, 2013).
4.2 Exploratory Data Analysis (EDA)
The EDA phase focused on identifying patterns within the dataset:
Distribution Analysis: The distribution of key variables like severity, complexity, and vendor_project was examined to spot trends and outliers, assessing the frequency of vulnerabilities across different severity levels and vendors.

Correlation Analysis: Correlations between variables such as severity, CVSS score, vector, and complexity were analyzed to uncover interdependencies and understand how characteristics like attack vector or complexity influenced vulnerability severity.

Visualizations: Visual representations, including heatmaps and bar charts, were used to display the distribution of vulnerabilities across products, vendors, and timeframes, highlighting risk concentrations and vulnerability spikes over time.

4.3 Bayesian Network Analysis
To model the relationships between variables, a Bayesian network was constructed:
Nodes Representation: Key dataset variables like severity, vector, complexity, and CVSS score were represented as nodes, illustrating the probabilistic relationships between them (Chockalingam et al., 2017).
Directed Arcs: Arcs indicated conditional dependencies, showing how variables such as CVSS score and vector influenced vulnerability severity, mapping out the interconnections between key factors.
Parameter Estimation: Maximum Likelihood Estimation (MLE) was used to calculate conditional probability tables (CPTs), providing insight into likely outcomes, such as predicting vulnerability severity based on certain characteristics.
4.4 Naive Bayes Classification
A Naive Bayes classifier was applied to predict vulnerability severity:
Feature Selection: Important features like attack vector, complexity, CVSS score, vendor_project, and product type were selected based on their correlation with vulnerability severity.
Model Training: The classifier was trained on 80% of the data, with 20% held back for testing, ensuring the model could generalize effectively to new data.
Performance Evaluation: The model was evaluated using accuracy, precision, recall, and F1 score, achieving 91.04% accuracy, indicating reliable performance in classifying vulnerability severity for cybersecurity risk management.
 
5. Exploratory Data Analysis (EDA)
5.1 Distribution of Vulnerabilities by Product
The analysis showed that certain products were disproportionately impacted by vulnerabilities in 2022 (Allodi and Massacci, 2017). Windows had the highest count with 310 vulnerabilities, followed by the Chromium V8 Engine (95), Win32k (75), and Chrome (70), while iOS and IOS XE Software were also frequently targeted, indicating their appeal to attackers. This tells us that product with larger user based has higher reported vulnerabilities, as a result, products like windows Chromium V8, and Chrome should prioritize more towards applying for secured measures.

![Picture1](https://github.com/user-attachments/assets/f17d15ae-e6b3-41d7-ae77-efb81ce17001)

5.2 Severity Levels

A significant portion of vulnerabilities were categorized as high severity, with over 25% classified as critical. Medium and low severity vulnerabilities made up a smaller share but still represented serious risks, particularly when combined with easily exploitable vectors.

![Picture2](https://github.com/user-attachments/assets/eec6a061-28df-4c06-80de-c3e0a62553a0)

5.3 Complexity of Exploitation

Most vulnerabilities in the dataset were low complexity, meaning they could be exploited with relative ease (Xie et al., 2010). Although high complexity vulnerabilities were less common, they often involved more advanced attack methods, posing severe risks if successfully exploited.

![Picture3](https://github.com/user-attachments/assets/3f80e4e0-3712-4e80-879f-ac518eade80b)

5.4 Patching Time Analysis
The median patching time was 796 days, while the average patching time extended to 885.8 days, highlighting long exposure windows. Notably, some vulnerabilities were pre-patched, with the shortest patching time being -44 days, indicating they were fixed before public disclosure.
 
![Picture4](https://github.com/user-attachments/assets/fe939fb4-40b3-4917-af0f-0c4f41c8b7d7)

5.5 Vendor Analysis\
The vendors with the most reported vulnerabilities were Microsoft (850), Cisco (280), Apple (190), and Google (183). This emphasizes the critical need for ongoing collaboration with these vendors to ensure timely vulnerability resolution and mitigate risks effectively.

![Picture5](https://github.com/user-attachments/assets/13ed2f20-286a-40c5-9e1e-b6916139e9b7)
 
6.1 Result and Analysis
In this report, Bayesian Network is used to look into how different variables of the cybersecurity vulnerability dataset are interconnected to one another such that they influence change in each other. In short, the Bayesian Network acts like a map for various characteristics and their relations with one another. With the help of this we can understand how certain features effect one another which can be leveraged to make certain prediction or derive of a certain scenario occurring based on the data obtained. Further, each of the "nodes" in the Bayesian Network is a feature and the arrows represent how each feature are linked or influenced with one another. Following figure shows the Bayesian Network visualization for various nodes and their relationship with each other through the help of nodes (Chockalingam et al., 2017).

![Picture6](https://github.com/user-attachments/assets/3ee24c21-1ec5-4865-824d-1ebd84b13884)

Based on the above figure, we can identify the following variables and their relationships:
Vendor and Product Influence: In the above figure, we can clearly notice that the node "vendor_project" affects various other nodes represented by the arrows stretched to other nodes, as a result, it can be claimed to be highly important in this network. In simpler terms, in this system the company and organization play a vital role in predicting the risk their product might have. This was also analyzed in the EDA process where a bar graph showed Microsoft which is a vendor had the highest number of vulnerabilities at 850. Moreover, the product node also stretched out arrows to other nodes suggesting it also has an influential factor over other factors such as ways to exploit vulnerability, complexity of attack, and risk seriousness. Further, this makes perfect sense as every product has its own strength and weakness as refereed to what they are design for.

Severity as a Dependent Variable: In any dataset related to cybersecurity, the most crucial feature is severity which talks about how harmful the vulnerabilities are. Further, form the figure, we can see that severity nodes' arrow extends to other features suggesting it depends on attack's nature, difficulty in exploit, and the CVSS score. As we can notice, that the severity is closely tied with another node named vector which talks about how the attack happens for instance, over a network, or on the same machine. Similarly, complexity is another node where the severity is dependent, which talks about the nature of attack if it was simple or complex. Finally, common vulnerability scoring system (CVSS) which is a score based on the risky of vulnerability, higher the score, higher dangerous level. 

Patching and Vulnerability Management: Likewise, from the above figure, it is clear that due_date, date_added and required_action are closely connected with each, meaning the action required to fix a vulnerability and the urgency for the problem to be resolve has a huge influence towards the deadline. Further, date added refers to the date in which a vulnerability is added, which is depended towards the due data as the problem will be fix as soon as the vulnerability is added to the system, this dependency helps to identify the urgent ones such that they are dealt first. 

Interaction Between Patching and Publishing: Here the pub_date refers to a specific date where a certain vulnerability that are made public. This node is closely connected with severity, due_date, and required_action. This close connection between all of this node refers to the fact that certain vulnerabilities which are serious in nature and require urgent action are published to the public much faster. This shows the urgency to fix vulnerabilities faster than they can be exploited. 
Now further, discussing about the network structure, various insights about how the vulnerabilities can be fix can be unrevealed. 
Role of Vendors in Vulnerabilities: The vendor or companies that manufacture the product have a very important role in showcasing how vulnerable the product might be. For example, 'windows' the product of Microsoft is highly vulnerable, but Microsoft being a reputed company it makes sure to always keep their customer's security on top. Hence, it is essential to communicate with the organization or to stay up to date with the companies update in order to fix any weaknesses. 
Insights into Vulnerability Severity: The Severity or the risk factor of any vulnerabilities are not determined by just one factors, rather things like difficulty to exploit, nature of attack, the score of severity plays major role in determining the Severity level (Mell and Scarfone, 2007). This understanding allows the security teams to know to look for all these factors while fixing an issue or tacking one.
Time Taken for Managing Vulnerability: As certain vulnerability can be more damaging than other; companies need to know which vulnerability to tackle first. In order to manage these vulnerabilities, details such as date of vulnerability discovery, deadline for fixing the problem, and action required should be considered.
Vulnerabilities Prediction: With the help of Bayesian Network, the understanding of severity of any new vulnerability can be easily identified with various factors discussed earlier which can be leveraged by organization to estimate severity levels quickly. 
While Bayesian Network has provided with various useful insights into dependencies of multiple factors, it is crucial to understand that, this model also consists of limitation which might affect the analysis process. For example, the model assumes that every feature remains the same over the period of time but in real-life this might not be applicable as new vulnerabilities might evolve requires different type of analysis. Also, the accuracy of the analysis is completely reliable on the quality of the data, if the data quality is compromised then the model’s output might not be accurate, as of which this report has emphasized on data quality highly (Kesan and Hayes, 2016). 

7. Social, Ethical, and Legal Considerations
At a more extensive level, consequences of this analysis underscore serious ethical as well as legal implications when vulnerabilities are not timely treated well. 
Dataset bias: There are some products and vendors that have more representation in the dataset than others. This can slightly affect the results towards products that are reviewed much more often by researchers or security services.
Privacy: The data that is used for the vulnerability analysis usually contains sensitive information and can be exploited if it falls into wrong hands. A key ethical issue is that this data should be strictly anonymized and stored securely (Sirur, Nurse and Webb, 2018).
Conclusion
The Bayesian network was useful to determine essential interconnections of vulnerability attributes. This allows security teams to prioritize their work — by severity, vector and complexity or all three factors combined which could result in a focus on only the top 10% area. The results shine a light on the importance of fast, effective patching mechanisms; failing to meet these benchmarks quickly leaves systems open to large amounts of risk.


Allodi, L. and Massacci, F., 2017. Attack potential in impact and complexity. In: Proceedings of the 12th International Conference on Availability, Reliability and Security, pp. 1-6.
Blakely, B., Kurtenbach, J. and Nowak, L., 2022. Exploring the information content of cyber breach reports and the relationship to internal controls. International Journal of Accounting Information Systems, 46, p.100568.
Chockalingam, S., Pieters, W., Teixeira, A. and van Gelder, P., 2017. Bayesian network models in cyber security: a systematic review. In: Secure IT Systems: 22nd Nordic Conference, NordSec 2017, Tartu, Estonia, November 8–10, 2017, Proceedings 22, pp. 105-122. Springer International Publishing.
Kesan, J.P. and Hayes, C.M., 2016. Bugs in the market: Creating a legitimate, transparent, and vendor-focused market for software vulnerabilities. Arizona Law Review, 58, p.753.
Mallinckrodt, C.H., 2013. Preventing and treating missing data in longitudinal clinical trials: a practical guide. Cambridge University Press.
Mell, P. and Scarfone, K., 2007. Improving the common vulnerability scoring system. IET Information Security, 1(3), pp.119-127.
Petkova, L., 2021. Cybersecurity trends. Security & Future, 5(4), pp.137-140.
Rothman, T. and Rothman, T., 2020. Company C: Cybersecurity. In: Valuations of Early-Stage Companies and Disruptive Technologies: How to Value Life Science, Cybersecurity and ICT Start-ups, and their Technologies, pp.165-187.
Samtani, S., Abate, M., Benjamin, V. and Li, W., 2020. Cybersecurity as an industry: A cyber threat intelligence perspective. In: The Palgrave Handbook of International Cybercrime and Cyberdeviance, pp.135-154.
Sirur, S., Nurse, J.R. and Webb, H., 2018. Are we there yet? Understanding the challenges faced in complying with the General Data Protection Regulation (GDPR). In: Proceedings of the 2nd International Workshop on Multimedia Privacy and Security, pp. 88-95.
Xie, P., Li, J.H., Ou, X., Liu, P. and Levy, R., 2010. Using Bayesian networks for cyber security analysis. In: 2010 IEEE/IFIP International Conference on Dependable Systems & Networks (DSN), pp. 211-220.

