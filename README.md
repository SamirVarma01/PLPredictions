# PLPredictions
A model that can predict the outcome of Premier League matches, utilizing data from the 1993-1994 season to November 2024.

Development Journey
Initial Approach
The platform began with a basic implementation using LightGBM for prediction:
Attempted to create rolling statistics for team performance
Initial attempt failed due to dataset column name mismatches
Identified need for proper feature engineering
Feature Engineering Evolution
1. Basic Model Implementation (F1: 0.6007)
Team performance metrics
Basic head-to-head statistics
Rolling averages for:
Goals scored
Shots on target
Team form
2. Enhanced Feature Set (F1: 0.6046)
Added comprehensive head-to-head historical statistics
Incorporated detailed win/loss/draw statistics
Improved granularity of performance metrics
