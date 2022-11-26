##  Use this repo for other projets

We recommend the use of `git submodule` as a way of including the integrators defined in this repo in your project.

consider the following [link](https://git-scm.com/book/en/v2/Git-Tools-Submodules) for more information about git submodules.

The following are a summary of commands to use:
* step 1: add the submodule to your project
```commandline
git submodule add https://github.com/mk-95-pycfd/integrators
```
* step 2: check that the submodule was added 
```
git status 
```
you should see 
```
On branch master
Your branch is up-to-date with 'origin/master'.

Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

	new file:   .gitmodules
	new file:   integrators
```

* step 3: commit the changes
```commandline
git commit -am 'Add integrators module'
```

* step 4: push the changes to the online repo
```commandline
git push origin master
```