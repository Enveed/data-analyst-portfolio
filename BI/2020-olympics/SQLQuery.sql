-- REVIEW ENTIRE DATASET
SELECT TOP (50) * FROM Portfolio..Athletes;

SELECT TOP (50) * FROM Portfolio..Coaches;

SELECT TOP (50) * FROM Portfolio..EntriesGender;

SELECT TOP (50) * FROM Portfolio..Medals;

SELECT TOP (50) * FROM Portfolio..Teams;

-- DATA EXPLORATION

-- Seeing the amount of athletes from each Country/NOC
SELECT NOC, COUNT(*) AS Total FROM Portfolio..Athletes GROUP BY NOC ORDER BY Total DESC;

-- Seeing the amount of coaches from each Country/NOC
SELECT NOC, COUNT(*) AS Total FROM Portfolio..Coaches GROUP BY NOC ORDER BY Total DESC;

-- Seeing the total amount of medals
SELECT SUM(CAST(Total AS int)) AS TotalMedals FROM Portfolio..Medals;

-- Seeing the total amount of attendants
SELECT SUM(CAST(Total AS int)) AS TotalAthletes FROM Portfolio..EntriesGender;

-- Checking for Cambodia attendants
SELECT * FROM (
	SELECT *, 'Athlete' AS Type FROM Portfolio..Athletes
	UNION
	SELECT Name, NOC, Discipline, 'Coach' AS Type FROM Portfolio..Coaches
) AS CamboCheck WHERE NOC = 'Cambodia';

-- Medals win percentage by team
SELECT 
	"Team/NOC" AS NOC, Gold, Silver, Bronze,  
	CAST(Total AS Int) * 100.0 / SUM(CAST(Total AS INT)) OVER () AS MedalPercentage
FROM Portfolio..Medals ORDER BY MedalPercentage DESC;

-- DATA EXPORT FOR VISUALIZATION
-- For AttendantsQuery.csv
SELECT *, 'Athlete' AS Type FROM Portfolio..Athletes
	UNION
SELECT Name, NOC, Discipline, 'Coach' AS Type FROM Portfolio..Coaches;

-- For GenderQuery.csv
SELECT Discipline, Female, Male FROM Portfolio..EntriesGender;

-- For MedalsQuery.csv
SELECT "Team/NOC" AS NOC, Gold, Silver, Bronze FROM Portfolio..Medals;

SELECT COUNT(distinct "Team/NOC") FROM Portfolio..Medals;

SELECT COUNT(distinct NOC) FROM (
SELECT *, 'Athlete' AS Type FROM Portfolio..Athletes
	UNION
SELECT Name, NOC, Discipline, 'Coach' AS Type FROM Portfolio..Coaches
) AS new;