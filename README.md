# IBMZDataHackathonProject
Repository for files related to IBM Z Datathon 2024.

Our group (ICAero), a group of aeronautics students from Imperial College London, designed and implemented a machine learning model. The object of our project was to design a piece of software to help decision-makers
and stakeholders in choosing optimal sites for renewable energy infrastructure construction. We initially focussed specifically on solar energy and solar farms, although there is much scope for broadening to other 
energy kinds.

We took into consideration 9 parameters, from land surface temperature, solar irradiation, albedo effect and many more. Using these, we collected data from a variety of sources through API requests to construct our dataset.
After partitioning the dataset into training and test, we then trained an MLP model to identify optimal sites. The suitability of each site was given on a normalised scale from 0 to 1, with 0 being suitable and 1 being best
suited.

We also designed a front-end UI for ease of use and accessibility. Our intention with the tool is to guide local authorities and policymakers with their decision-making. We sincerely hope that we have achieved this goal to
the best of our ability. 

From the ICAero Team
