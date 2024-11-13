# PLPredictions
A model that can predict the outcome of Premier League matches, utilizing data from the 1993-1994 season to November 2024.

Feature Engineering Evolution
1. Basic Model Implementation (F1: 0.6007)
Team performance metrics
- Basic head-to-head statistics
- Rolling averages for:
- Goals scored
- Shots on target
- Team form
2. Enhanced Feature Set (F1: 0.6046)
- Added comprehensive head-to-head historical statistics
- Incorporated detailed win/loss/draw statistics
- Improved granularity of performance metrics
3. Venue-Specific Analysis (F1: 0.5986)
- Split features into home/away specific metrics Implemented separate tracking for:
- Home game performance
- Away game performance
- Venue-specific form calculations
4. Additional Factors (F1: 0.4724)
- Incorporated referee impact through encoding Simplified feature set for better generalization
- Implemented time-based validation approach
- Best Performing Model (F1: 0.8326)
- Key Features
    Team streak statistics Win streaks
    Loss streaks
    Unbeaten runs
    Historical head-to-head performance
    Recent form using rolling averages
   Venue-specific performance metrics
  Time series cross-validation
Technical Learnings Critical Success Factors
1. Feature Importance
Team form proved crucial for prediction
Head-to-head statistics significantly improved accuracy Venue-specific performance metrics added valuable context
2. Data Handling
Careful management of temporal aspects Prevention of data leakage
Proper handling of rolling statistics
3. Model Improvements
Automatic identification and correction of column mismatches
Implementation of proper index handling Optimization of feature calculation timing
Conclusion
Through automated iteration and improvement, the platform successfully developed a robust model for Premier League match prediction. The final model demonstrates strong predictive capabilities with an F1 score of 0.8326, effectively capturing the complex patterns in football match outcomes.
