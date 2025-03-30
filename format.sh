#!/usr/bin/env bash

ruff check --fix citl
ruff format citl
ruff check --fix test
ruff format test
