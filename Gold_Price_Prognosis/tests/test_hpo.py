from gold_forecasting.hpo import run_study

def test_study_resumes_only_missing_trials(tmp_path):
    calls = []
    def objective(trial):
        value = trial.suggest_float("x", 0, 1)
        calls.append(value)
        return (value - 0.5) ** 2
    run_study("resume_test", objective, 2, tmp_path, seed=1)
    assert len(calls) == 2
    run_study("resume_test", objective, 5, tmp_path, seed=1)
    assert len(calls) == 5  # 2 already-completed trials were reused; only the 3 missing ones ran
