-- Tests for evalDynFAE and evalStatFAE.  test2 should demonstrate
-- the difference between static and dynamic scoping.  If you get the same
-- results with both interpreters, you've got problems.

test0=(App (Lambda "inc" (Id "inc")) (Lambda "x" (Plus (Id "x") (Num 1))))
test1=(App (Lambda "inc" (App (Id "inc") (Num 3))) (Lambda "x" (Plus (Id "x") (Num 1))))
test2=(App (Lambda "n" (App (Lambda "inc" (App (Lambda "n" (App (Id "inc") (Num 3))) (Num 3))) (Lambda "x" (Plus (Id "x") (Id "n"))))) (Num 1))

-- List of tests if you would like to use map for testing

tests = [test0,test1,test2]

-- Tests for evalFBAE.  These are the same tests as above
-- using Bind.

test0X= (BindD "inc" (LambdaD "x" (PlusD (IdD "x") (NumD 1))) (IdD "inc"))
test1X = (BindD "inc" (LambdaD "x" (PlusD (IdD "x") (NumD 1))) (AppD (IdD "inc") (NumD 3)))
test2X = (BindD "n" (NumD 1) (BindD "inc" (LambdaD "x" (PlusD (IdD "x") (IdD "n"))) (BindD "n" (NumD 3) (AppD (IdD "inc") (NumD 3)))))

-- List of tests if you would like to use map for testing

testsX = [test0X,test1X,test2X]


-- Tests for evalFBAEC.  These are the same tests as above
-- using the AST for FBAEC.

test0XX= (BindE "inc" (LambdaE "x" (PlusE (IdE "x") (NumE 1))) (IdE "inc"))
test1XX = (BindE "inc" (LambdaE "x" (PlusE (IdE "x") (NumE 1))) (AppE (IdE "inc") (NumE 3)))
test2XX = (BindE "n" (NumE 1) (BindE "inc" (LambdaE "x" (PlusE (IdE "x") (IdE "n"))) (BindE "n" (NumE 3) (AppE (IdE "inc") (NumE 3)))))

-- List of tests if you would like to use map for testing

testsXX = [test0XX,test1XX,test2XX]
