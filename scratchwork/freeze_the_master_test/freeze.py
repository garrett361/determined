import workspaces

workspace = workspaces.Workspace(workspace_name="Ensembling")
all_trials = workspace.get_all_trials()
print(f"Number of trials in workspace: {len(all_trials)}")
