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
--evalDynFAE _ _ = Nothing

data FAEValue where
  NumV :: Int -> FAEValue
  ClosureV :: String -> FAE -> Env' -> FAEValue
  deriving (Show,Eq)
  
type Env' = [(String,FAEValue)]

evalStatFAE :: Env' -> FAE -> (Maybe FAEValue)
evalStatFAE _ _ = Nothing


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
elabFBAE _ = (Num (-1))

evalFBAE :: Env' -> FBAE -> (Maybe FAEValue)
evalFBAE _ _ = Nothing

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


run_test_cases = do
      putStrLn "Running test cases..."
      print "evalDynFAE test cases"
      print ("evalDynFAE [] (Num 5): ", evalDynFAE_t1)
      print ("evalDynFAE [] (Plus (Num 5) (Num 7)): ", evalDynFAE_t2)
      print ("evalDynFAE [] (Minus (Num 5) (Num 3)): ", evalDynFAE_t3)
      print ("evalDynFAE [] (Lambda 'x' ('x' + 1)): ", evalDynFAE_t4)
      print ("evalDynFAE [] (App Lambda 'x' (Plus (Id 'x') (Num 1)) (Num 6)): ", evalDynFAE_t5)


main = do run_test_cases
