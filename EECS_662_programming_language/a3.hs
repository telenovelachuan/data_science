{-# LANGUAGE GADTs #-}

-- Imports for Monads

import Control.Monad

-- FAE AST and Type Definitions

data FAE where
  Num :: Int -> FAE
  Plus :: FAE -> FAE -> FAE
  Minus :: FAE -> FAE -> FAE
  Lambda :: String -> FAE -> FAE
  App :: FAE -> FAE -> FAE
  Id :: String -> FAE
  deriving (Show,Eq)

type Env = [(String,FAE)]

evalDynFAE :: Env -> FAE -> (Maybe FAE)
evalDynFAE e (Id s) = lookup s e
evalDynFAE e (Num n) = Just (Num n)
evalDynFAE e (Plus a b) = do {
  (Num a') <- evalDynFAE e a;
  (Num b') <- evalDynFAE e b;
  return (Num (a' + b'))
}
evalDynFAE e (Minus a b) = do {
  (Num a') <- evalDynFAE e a;
  (Num b') <- evalDynFAE e b;
  if a' >= b' then return (Num (a' - b')) else Nothing
}
evalDynFAE e (Lambda i s) = Just (Lambda i s)
evalDynFAE e (App f a) = do {
  a' <- evalDynFAE e a;
  (Lambda i s) <- evalDynFAE e f;
  (evalDynFAE ((i,a'):e) s);
}


data FAEValue where
  NumV :: Int -> FAEValue
  ClosureV :: String -> FAE -> Env' -> FAEValue
  deriving (Show,Eq)
  
type Env' = [(String,FAEValue)]

evalStatFAE :: Env' -> FAE -> (Maybe FAEValue)
evalStatFAE e (Num n) = Just (NumV n)
evalStatFAE e (Id s) = lookup s e
--evalStatFAE e (ClosureV )
evalStatFAE e (Plus a b) = do {
  (NumV a') <- evalStatFAE e a;
  (NumV b') <- evalStatFAE e b;
  return (NumV (a' + b'))
}
evalStatFAE e (Minus a b) = do {
  (NumV a') <- evalStatFAE e a;
  (NumV b') <- evalStatFAE e b;
  if a' >= b' then return (NumV (a' - b')) else Nothing
}
evalStatFAE e (Lambda i s) = return (ClosureV i s e)
evalStatFAE e (App f a) = do {
  a' <- evalStatFAE e a;
  (ClosureV i s e) <- evalStatFAE e f;
  (evalStatFAE ((i, a'):e) s);
}


-- FBAE AST and Type Definitions

data FBAE where
  NumD :: Int -> FBAE
  PlusD :: FBAE -> FBAE -> FBAE
  MinusD :: FBAE -> FBAE -> FBAE
  LambdaD :: String -> FBAE -> FBAE
  AppD :: FBAE -> FBAE -> FBAE
  BindD :: String -> FBAE -> FBAE -> FBAE
  IdD :: String -> FBAE
  deriving (Show,Eq)

elabFBAE :: FBAE -> FAE
elabFBAE (NumD n) = Num n
elabFBAE (IdD x) = Id x
elabFBAE (PlusD a b) = Plus (elabFBAE a) (elabFBAE b)
elabFBAE (MinusD a b) = Minus (elabFBAE a) (elabFBAE b)
elabFBAE (LambdaD i s) = Lambda i (elabFBAE s)
elabFBAE (AppD f a) = App (elabFBAE f) (elabFBAE a)
elabFBAE (BindD i a s) = App (Lambda i (elabFBAE s)) (elabFBAE a)


evalFBAE :: Env' -> FBAE -> (Maybe FAEValue)
evalFBAE e s = evalStatFAE e (elabFBAE s)


-- FBAEC AST and Type Definitions

data FBAEC where
  NumE :: Int -> FBAEC
  PlusE :: FBAEC -> FBAEC -> FBAEC
  MinusE :: FBAEC -> FBAEC -> FBAEC
  TrueE :: FBAEC
  FalseE :: FBAEC
  AndE :: FBAEC -> FBAEC -> FBAEC
  OrE :: FBAEC -> FBAEC -> FBAEC
  NotE :: FBAEC -> FBAEC
  IfE :: FBAEC -> FBAEC -> FBAEC -> FBAEC
  LambdaE :: String -> FBAEC -> FBAEC
  AppE :: FBAEC -> FBAEC -> FBAEC
  BindE :: String -> FBAEC -> FBAEC -> FBAEC
  IdE :: String -> FBAEC
  deriving (Show,Eq)

elabFBAEC :: FBAEC -> FAE
elabFBAEC _ = (Num (-1))

evalFBAEC :: Env' -> FBAEC -> Maybe FAEValue
evalFBAEC _ _ = Nothing


--evalDynFAE test cases
evalDynFAE_t1 = evalDynFAE [] (Num 5)
evalDynFAE_t2 = evalDynFAE [] (Plus (Num 5) (Num 7))
evalDynFAE_t3 = evalDynFAE [] (Minus (Num 5) (Num 3))
evalDynFAE_t4 = evalDynFAE [] (Lambda "x" (Plus (Id "x") (Num 1)))
evalDynFAE_t5 = evalDynFAE [] (App (Lambda "x" (Plus (Id "x") (Num 1))) (Num 6))
evalDynFAE_t6 = evalDynFAE [("x", (Num 2))] (Id "x")

--evalStatFAE test cases
evalStatFAE_t1 = evalStatFAE [] (Num 5)
evalStatFAE_t2 = evalStatFAE [] (Plus (Num 5) (Num 7))
evalStatFAE_t3 = evalStatFAE [] (Minus (Num 5) (Num 3))
evalStatFAE_t4 = evalStatFAE [] (Lambda "x" (Plus (Id "x") (Num 1)))
evalStatFAE_t5 = evalStatFAE [] (App (Lambda "x" (Plus (Id "x") (Num 1))) (Num 6))
evalStatFAE_t6 = evalStatFAE [("x", (NumV 2))] (Id "x")

--elabFBAE test cases
elabFBAE_t1 = elabFBAE (NumD 5)
elabFBAE_t2 = elabFBAE (PlusD (NumD 5) (NumD 7))
elabFBAE_t3 = elabFBAE (MinusD (NumD 5) (NumD 3))
elabFBAE_t4 = elabFBAE (LambdaD "x" (PlusD (IdD "x") (NumD 1)))
elabFBAE_t5 = elabFBAE (AppD (LambdaD "x" (PlusD (IdD "x") (NumD 1))) (NumD 6))
elabFBAE_t6 = elabFBAE (IdD "x")

--evalFBAE test cases
evalFBAE_t1 = evalFBAE [] (NumD 5)
evalFBAE_t2 = evalFBAE [] (PlusD (NumD 5) (NumD 7))
evalFBAE_t3 = evalFBAE [] (MinusD (NumD 5) (NumD 3))
evalFBAE_t4 = evalFBAE [] (LambdaD "x" (PlusD (IdD "x") (NumD 1)))
evalFBAE_t5 = evalFBAE [] (AppD (LambdaD "x" (PlusD (IdD "x") (NumD 1))) (NumD 6))
evalFBAE_t6 = evalFBAE [] (IdD "x")


run_test_cases = do
      putStrLn "Running test cases..."
      print "evalDynFAE test cases"
      print ("evalDynFAE [] (Num 5): ", evalDynFAE_t1)
      print ("evalDynFAE [] (Plus (Num 5) (Num 7)): ", evalDynFAE_t2)
      print ("evalDynFAE [] (Minus (Num 5) (Num 3)): ", evalDynFAE_t3)
      print ("evalDynFAE [] (Lambda 'x' (Plus (Id 'x') (Num 1))): ", evalDynFAE_t4)
      print ("evalDynFAE [] (App Lambda 'x' (Plus (Id 'x') (Num 1)) (Num 6)): ", evalDynFAE_t5)
      print ("evalDynFAE [('x', 2)] (Id 'x'): ", evalDynFAE_t6)

      print "evalStatFAE test cases"
      print ("evalStatFAE [] (Num 5): ", evalStatFAE_t1)
      print ("evalStatFAE [] (Plus (Num 5) (Num 7)): ", evalStatFAE_t2)
      print ("evalStatFAE [] (Minus (Num 5) (Num 3)): ", evalStatFAE_t3)
      print ("evalStatFAE [] (Lambda 'x' (Plus (Id 'x') (Num 1))): ", evalStatFAE_t4)
      print ("evalStatFAE [] (App Lambda 'x' (Plus (Id 'x') (Num 1)) (Num 6)): ", evalStatFAE_t5)
      print ("evalStatFAE [('x', 2)] (Id 'x'): ", evalStatFAE_t6)

      print "elabFBAE test cases"
      print ("elabFBAE (NumD 5): ", elabFBAE_t1)
      print ("elabFBAE (PlusD (NumD 5) (NumD 7)): ", elabFBAE_t2)
      print ("elabFBAE (MinusD (NumD 5) (NumD 3)): ", elabFBAE_t3)
      print ("elabFBAE elabFBAE (LambdaD 'x' (PlusD (IdD 'x') (NumD 1))): ", elabFBAE_t4)
      print ("elabFBAE (AppD (LambdaD 'x' (PlusD (IdD 'x') (NumD 1))) (NumD 6)): ", elabFBAE_t5)
      print ("elabFBAE (IdD 'x'): ", elabFBAE_t6)

      print "evalFBAE test cases"
      print ("evalFBAE [] (NumD 5): ", evalFBAE_t1)
      print ("evalFBAE [] (PlusD (NumD 5) (NumD 7)): ", evalFBAE_t2)
      print ("evalFBAE [] (MinusD (NumD 5) (NumD 3)): ", evalFBAE_t3)
      print ("evalFBAE [] elabFBAE (LambdaD 'x' (PlusD (IdD 'x') (NumD 1))): ", evalFBAE_t4)
      print ("evalFBAE [] (AppD (LambdaD 'x' (PlusD (IdD 'x') (NumD 1))) (NumD 6)): ", evalFBAE_t5)
      print ("evalFBAE [] (IdD 'x'): ", evalFBAE_t6)


main = do run_test_cases
