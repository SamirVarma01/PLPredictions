import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")

# Load and prepare data
df = pd.read_csv("./input/PremierLeague.csv")
df["Date"] = pd.to_datetime(df["Date"])


def get_streak_stats(results):
    """Calculate streak-based statistics from recent results."""
    if not results:
        return {
            "win_streak": 0,
            "loss_streak": 0,
            "unbeaten_streak": 0,
            "winless_streak": 0,
        }

    current_win_streak = 0
    current_loss_streak = 0
    current_unbeaten_streak = 0
    current_winless_streak = 0

    # Calculate current streaks
    for result in results:
        if result == 1:  # Win
            current_win_streak += 1
            current_loss_streak = 0
            current_unbeaten_streak += 1
            current_winless_streak = 0
        elif result == -1:  # Loss
            current_win_streak = 0
            current_loss_streak += 1
            current_unbeaten_streak = 0
            current_winless_streak += 1
        else:  # Draw
            current_win_streak = 0
            current_loss_streak = 0
            current_unbeaten_streak += 1
            current_winless_streak += 1

    return {
        "win_streak": current_win_streak,
        "loss_streak": current_loss_streak,
        "unbeaten_streak": current_unbeaten_streak,
        "winless_streak": current_winless_streak,
    }


def get_head_to_head_stats(data, home_team, away_team, date, n_matches=5):
    """Get historical head-to-head statistics between two teams."""
    mask = (
        ((data["HomeTeam"] == home_team) & (data["AwayTeam"] == away_team))
        | ((data["HomeTeam"] == away_team) & (data["AwayTeam"] == home_team))
    ) & (data["Date"] < date)

    previous_matches = data[mask].sort_values("Date", ascending=False).head(n_matches)

    if len(previous_matches) == 0:
        return {"h2h_wins": 0, "h2h_goals_scored": 0, "h2h_goals_conceded": 0}

    wins = 0
    goals_scored = 0
    goals_conceded = 0

    for _, match in previous_matches.iterrows():
        if match["HomeTeam"] == home_team:
            goals_scored += match["FullTimeHomeTeamGoals"]
            goals_conceded += match["FullTimeAwayTeamGoals"]
            if match["FullTimeResult"] == "H":
                wins += 1
        else:
            goals_scored += match["FullTimeAwayTeamGoals"]
            goals_conceded += match["FullTimeHomeTeamGoals"]
            if match["FullTimeResult"] == "A":
                wins += 1

    n_matches_actual = len(previous_matches)
    return {
        "h2h_wins": wins / n_matches_actual,
        "h2h_goals_scored": goals_scored / n_matches_actual,
        "h2h_goals_conceded": goals_conceded / n_matches_actual,
    }


def create_team_features(data, n_matches=5):
    team_stats = {}

    # Add head-to-head features
    data["h2h_wins"] = 0
    data["h2h_goals_scored"] = 0
    data["h2h_goals_conceded"] = 0

    # Add streak features
    streak_features = ["win_streak", "loss_streak", "unbeaten_streak", "winless_streak"]
    for feature in streak_features:
        data[f"home_{feature}"] = 0
        data[f"away_{feature}"] = 0

    # Statistics to track
    cols_to_track = [
        "FullTimeHomeTeamGoals",
        "FullTimeAwayTeamGoals",
        "HomeTeamShots",
        "AwayTeamShots",
        "HomeTeamShotsOnTarget",
        "AwayTeamShotsOnTarget",
    ]

    for idx, row in data.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        # Get head-to-head stats
        h2h_stats = get_head_to_head_stats(data, home_team, away_team, row["Date"])
        data.loc[idx, "h2h_wins"] = h2h_stats["h2h_wins"]
        data.loc[idx, "h2h_goals_scored"] = h2h_stats["h2h_goals_scored"]
        data.loc[idx, "h2h_goals_conceded"] = h2h_stats["h2h_goals_conceded"]

        if home_team not in team_stats:
            team_stats[home_team] = {
                "recent_goals": [],
                "recent_shots": [],
                "recent_shots_target": [],
                "form": [],
            }
        if away_team not in team_stats:
            team_stats[away_team] = {
                "recent_goals": [],
                "recent_shots": [],
                "recent_shots_target": [],
                "form": [],
            }

        # Get streak stats for both teams
        home_streak_stats = get_streak_stats(team_stats[home_team]["form"])
        away_streak_stats = get_streak_stats(team_stats[away_team]["form"])

        # Update streak features
        for feature in streak_features:
            data.loc[idx, f"home_{feature}"] = home_streak_stats[feature]
            data.loc[idx, f"away_{feature}"] = away_streak_stats[feature]

        # Update statistics for both teams
        if len(team_stats[home_team]["recent_goals"]) >= n_matches:
            team_stats[home_team]["recent_goals"] = team_stats[home_team][
                "recent_goals"
            ][1:]
            team_stats[home_team]["recent_shots"] = team_stats[home_team][
                "recent_shots"
            ][1:]
            team_stats[home_team]["recent_shots_target"] = team_stats[home_team][
                "recent_shots_target"
            ][1:]
            team_stats[home_team]["form"] = team_stats[home_team]["form"][1:]

        if len(team_stats[away_team]["recent_goals"]) >= n_matches:
            team_stats[away_team]["recent_goals"] = team_stats[away_team][
                "recent_goals"
            ][1:]
            team_stats[away_team]["recent_shots"] = team_stats[away_team][
                "recent_shots"
            ][1:]
            team_stats[away_team]["recent_shots_target"] = team_stats[away_team][
                "recent_shots_target"
            ][1:]
            team_stats[away_team]["form"] = team_stats[away_team]["form"][1:]

        # Add current match statistics
        team_stats[home_team]["recent_goals"].append(row["FullTimeHomeTeamGoals"])
        team_stats[home_team]["recent_shots"].append(row["HomeTeamShots"])
        team_stats[home_team]["recent_shots_target"].append(
            row["HomeTeamShotsOnTarget"]
        )

        team_stats[away_team]["recent_goals"].append(row["FullTimeAwayTeamGoals"])
        team_stats[away_team]["recent_shots"].append(row["AwayTeamShots"])
        team_stats[away_team]["recent_shots_target"].append(
            row["AwayTeamShotsOnTarget"]
        )

        # Update form
        if row["FullTimeResult"] == "H":
            team_stats[home_team]["form"].append(1)
            team_stats[away_team]["form"].append(-1)
        elif row["FullTimeResult"] == "A":
            team_stats[home_team]["form"].append(-1)
            team_stats[away_team]["form"].append(1)
        else:
            team_stats[home_team]["form"].append(0)
            team_stats[away_team]["form"].append(0)

        # Calculate features for current row
        data.loc[idx, "home_goals_avg"] = (
            np.mean(team_stats[home_team]["recent_goals"])
            if team_stats[home_team]["recent_goals"]
            else 0
        )
        data.loc[idx, "away_goals_avg"] = (
            np.mean(team_stats[away_team]["recent_goals"])
            if team_stats[away_team]["recent_goals"]
            else 0
        )
        data.loc[idx, "home_shots_avg"] = (
            np.mean(team_stats[home_team]["recent_shots"])
            if team_stats[home_team]["recent_shots"]
            else 0
        )
        data.loc[idx, "away_shots_avg"] = (
            np.mean(team_stats[away_team]["recent_shots"])
            if team_stats[away_team]["recent_shots"]
            else 0
        )
        data.loc[idx, "home_shots_target_avg"] = (
            np.mean(team_stats[home_team]["recent_shots_target"])
            if team_stats[home_team]["recent_shots_target"]
            else 0
        )
        data.loc[idx, "away_shots_target_avg"] = (
            np.mean(team_stats[away_team]["recent_shots_target"])
            if team_stats[away_team]["recent_shots_target"]
            else 0
        )
        data.loc[idx, "home_form"] = (
            np.mean(team_stats[home_team]["form"])
            if team_stats[home_team]["form"]
            else 0
        )
        data.loc[idx, "away_form"] = (
            np.mean(team_stats[away_team]["form"])
            if team_stats[away_team]["form"]
            else 0
        )

    return data


# Prepare features
df_processed = create_team_features(df.copy())

# Encode categorical variables
le = LabelEncoder()
df_processed["HomeTeam"] = le.fit_transform(df_processed["HomeTeam"])
df_processed["AwayTeam"] = le.fit_transform(df_processed["AwayTeam"])

# Create feature matrix
feature_cols = [
    "HomeTeam",
    "AwayTeam",
    "home_goals_avg",
    "away_goals_avg",
    "home_shots_avg",
    "away_shots_avg",
    "home_shots_target_avg",
    "away_shots_target_avg",
    "home_form",
    "away_form",
    "h2h_wins",
    "h2h_goals_scored",
    "h2h_goals_conceded",
    "home_win_streak",
    "home_loss_streak",
    "home_unbeaten_streak",
    "home_winless_streak",
    "away_win_streak",
    "away_loss_streak",
    "away_unbeaten_streak",
    "away_winless_streak",
]

X = df_processed[feature_cols]
y = df_processed["FullTimeResult"]

# Convert target to numeric
le_result = LabelEncoder()
y = le_result.fit_transform(y)

# Time series split for validation
tscv = TimeSeriesSplit(n_splits=5)

# Train and evaluate model
model = lgb.LGBMClassifier(random_state=42)
f1_scores = []

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="weighted")
    f1_scores.append(f1)

print(f"Average weighted F1 score: {np.mean(f1_scores):.4f}")
