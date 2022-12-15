This README.txt file was generated on 2022-06-29 by Thomas J. Yamashita


GENERAL INFORMATION

1. Title of Dataset: Distinguishing buildings from vegetation in an urban-chaparral mosaic landscape
	This data is associated with the manuscript, titled Distinguishing buildings from vegetation in an urban-chaparral mosaic landscape, available here: XXXX

2. Author Information
	Thomas J. Yamashita
		Caesar Kleberg Wildlife Research Institute, Texas A&M University - Kingsville
		tjyamashta@gmail.com
		Corresponding Author
	David B. Wester
		Caesar Kleberg Wildlife Research Institute, Texas A&M University - Kingsville
	Michael E. Tewes
		Caesar Kleberg Wildlife Research Institute, Texas A&M University - Kingsville
	John H. Young Jr. 
		Environmental Affairs Division, Texas Department of Transportation
	Jason V. Lombardi
		Caesar Kleberg Wildlife Research Institute, Texas A&M University - Kingsville

3. Date of Data Collection: 
	LiDAR data collected in Fall/Winter 2018

4. Geographic location of data collection: Eastern Cameron County, Texas, USA around State Highway 100, Farm-to-Market 106, and Farm-to-Market 1847

5. Funding Sources: Texas Department of Transportation


DATA & FILE OVERVIEW

1. File List: 
	DiscriminantAnalysis_Accuracy.xlsx: An excel file with the subset of buildings and non-buildings used to assess accuracy of the discriminant function
	DiscriminantAnalysis_for_Publication.R: R code for running the discriminant function
	DiscriminantAnalysis_Full.xlsx: An excel file containing all potential buildings
	DiscriminantAnalysis_Polygons.shp: A shapefile containing the polygons for all buildings
	DiscriminantAnalysis_Testing.xlsx: An excel file with the subset of potential buildings used for testing the discriminant function
	DiscriminantAnalysis_Training.xlsx: An excel file with the subset of potential buildings use for training the discriminant function
	README.txt: This file

2. Relationship between files: 
	The ID column in all datasets (xlsx and shp files) are linked. A row in the DiscriminantAnalysis_Training.xlsx file with an ID of 25 is the same ID as the polygon with ID 25 in the DiscriminantAnalysis_Polygons.shp file


METHODOLOGICAL INFORMATION

1. Description of the methods used for collection/generation and processing of data: 
	Methodology for collection and processing of the data can be found in the manuscript

2. Quality Assurance Procedures: 
	Quality assurance is discussed in the manuscript. Building identification was 95% accurate

3. People involved with data collection, processing, and analysis: 
	Thomas J. Yamashita, David B. Wester, Jason V. Lombardi


DATA SPECIFIC INFORMATION FOR: DiscriminantAnalysis_Accuracy.xlsx
1. Data Type: Microsoft Excel File

2. Number of Variables: 6

3. Number of Rows: 1000

4. Variable List: 
	ID: Unique Identifier for each individual polygon. Consistent across all files
	UTM_X_ctr: The X coordinate of the polygon centerpoint in the NAD83, UTM Zone 14N coordinate system
	UTM_Y_ctr: The Y coordinate of the polygon centerpoint in the NAD83, UTM Zone 14N coordinate system
	Area_m2: The area in square meters of the polygon
	class: The class that the discriminant function classified each row into (y=building, n=non-building)
	observed: The observed class of each row. This was assessed manually


DATA SPECIFIC INFORMATION FOR: DiscriminantAnalysis_for_Publication.R
1. Data Type: R script

5. Other Information: This script provides code for performing and assessing the discriminant function by hand and using the qda function in the MASS package
	


DATA SPECIFIC INFORMATION FOR: DiscriminantAnalysis_Full.xlsx
1. Data Type: Microsoft Excel File

2. Number of Variables: 12

3. Number of Rows: 49553

4. Variable List: 
	ID: Unique Identifier for each individual polygon. Consistent across all files
	Count_Total: The total number of LiDAR points in each polygon
	Count_1: The number of points for the Unclassified class
	Count_2: The number of points for the Ground class
	Count_6: The number of points for the Building class
	Count_7: The number of points for the Low Noise class
	Count_9: The number of points for the Water class
	Count_10: The number of points for the Rail class
	Count_17: The number of points for the Bridge Deck class
	Count_18: The number of points for the High Noise class
	Count_64: The number of points for a User-Defined class of points that were classified as buildings by the Planar Point Filter but excluded in the Point Tracing and Squaring tasks in LP360
	Building: Whether the polygon was classified as a building or non-building. This was intentionally left blank for the full dataset


DATA SPECIFIC INFORMATION FOR: DiscriminantAnalysis_Polygons.shp
1. Data Type: Shapefile

2. Number of Variables: 5

3. Number of Rows: 49553

4. Variable List: 
	FID: The ArcGIS associated ID for the shapefile that this was derived from
	Shape: ArcGIS mandatory field describing the shape of the item
	ID: Unique Identifier for each individual polygon. Consistent across all files
	Shape_Length: ArcGIS mandatory field describing the length of the perimeter of each polygon
	Shape_Area: ArcGIS mandatory field describing the area of each polygon


DATA SPECIFIC INFORMATION FOR: DiscriminantAnalysis_Testing.xlsx
1. Data Type: Microsoft Excel File

2. Number of Variables: 12

3. Number of Rows: 500

4. Variable List: 
	ID: Unique Identifier for each individual polygon. Consistent across all files
	Count_Total: The total number of LiDAR points in each polygon
	Count_1: The number of points for the Unclassified class
	Count_2: The number of points for the Ground class
	Count_6: The number of points for the Building class
	Count_7: The number of points for the Low Noise class
	Count_9: The number of points for the Water class
	Count_10: The number of points for the Rail class
	Count_17: The number of points for the Bridge Deck class
	Count_18: The number of points for the High Noise class
	Count_64: The number of points for a User-Defined class of points that were classified as buildings by the Planar Point Filter but excluded in the Point Tracing and Squaring tasks in LP360
	Building: Whether the polygon was classified as a building or non-building


DATA SPECIFIC INFORMATION FOR: DiscriminantAnalysis_Training.xlsx
1. Data Type: Microsoft Excel File

2. Number of Variables: 12

3. Number of Rows: 500

4. Variable List: 
	ID: Unique Identifier for each individual polygon. Consistent across all files
	Count_Total: The total number of LiDAR points in each polygon
	Count_1: The number of points for the Unclassified class
	Count_2: The number of points for the Ground class
	Count_6: The number of points for the Building class
	Count_7: The number of points for the Low Noise class
	Count_9: The number of points for the Water class
	Count_10: The number of points for the Rail class
	Count_17: The number of points for the Bridge Deck class
	Count_18: The number of points for the High Noise class
	Count_64: The number of points for a User-Defined class of points that were classified as buildings by the Planar Point Filter but excluded in the Point Tracing and Squaring tasks in LP360
	Building: Whether the polygon was classified as a building or non-building

