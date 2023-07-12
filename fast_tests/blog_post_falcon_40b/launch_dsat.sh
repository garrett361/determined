#!/usr/bin/env bash
python3 -m determined.pytorch.dsat asha falcon_40B_blog_post.yaml . -start_profile-step 10 -end-profile-step 20
