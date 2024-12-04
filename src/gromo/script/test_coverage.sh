#!/bin/sh

echo "Run Unit Tests with Coverage"

coverage run -m unittest
coverage xml
coverage html
echo "Done"
