{-# LANGUAGE GADTs,FlexibleContexts #-}

-- Imports for Monads

import Control.Monad

-- BBAE AST and Type Definitions

data TBBAE where
  TNum :: TBBAE
  TBool :: TBBAE
  deriving (Show,Eq)

data BBAE where
  Num :: Int -> BBAE
  Plus :: BBAE -> BBAE -> BBAE
  Minus :: BBAE -> BBAE -> BBAE
  Bind :: String -> BBAE -> BBAE -> BBAE
  Id :: String -> BBAE
  Boolean :: Bool -> BBAE
  And :: BBAE -> BBAE -> BBAE
  Leq :: BBAE -> BBAE -> BBAE
  IsZero :: BBAE -> BBAE
  If :: BBAE -> BBAE -> BBAE -> BBAE
  deriving (Show,Eq)

type Env = [(String,BBAE)]

type Cont = [(String,TBBAE)]

subst :: String -> BBAE -> BBAE ->BBAE
subst x v (Num n) = (Num n)
subst x v (Plus l r) = (Plus (subst x v l) (subst x v r))
subst x v (Minus l r) = (Minus (subst x v l) (subst x v r))
subst x v (Bind x' v' b') = (Bind x' (subst x v v') (subst x v b'))
subst x v (Id x') = if x == x' then v else (Id x')

evalS :: BBAE -> (Maybe BBAE)
evalS (Num x) = Just (Num x)
evalS (Boolean x) = Just (Boolean x)
evalS (Plus l r) = do {
  (Num l') <- evalS l;
  (Num r') <- evalS r;
  Just (Num (l' + r'))
}
evalS (Minus l r) = do {
  (Num l') <- evalS l;
  (Num r') <- evalS r;
  if l' >= r' then return (Num (l' - r')) else Nothing
}
evalS (Bind x a b) = do {
  (Num a') <- evalS a;
  (evalS (subst x (Num a') b))
}
evalS (Id x) = Nothing
evalS (And l r) = do {
  (Boolean l') <- evalS l;
  (Boolean r') <- evalS r;
  return (Boolean (l' && r'))
}
evalS (Leq l r) = do {
  (Num l') <- evalS l;
  (Num r') <- evalS r;
  if l' <=0 || r' <= 0 then Nothing else return (Boolean (l' <= r'))
}
evalS (IsZero x) = do {
  (Num x') <- evalS x;
  if x' < 0 then Nothing else return (Boolean (x' == 0))
}
evalS (If a b c) = do {
  (Boolean a') <- evalS a;
  if a' then (evalS b) else (evalS c)
}

--evalS test cases
evalS_t1 = evalS (Num 5)
evalS_t2 = evalS (Plus (Num 5) (Num 6))
evalS_t3 = evalS (Minus (Num 6) (Num 5))
evalS_t4 = evalS (Minus (Num 5) (Num 6))
evalS_t5 = evalS (Boolean True)
evalS_t6 = evalS (Bind "x" (Num 3) (Plus (Num 4) (Id "x")))
evalS_t7 = evalS (Leq (Num 4) (Num 5))
evalS_t8 = evalS (Leq (Num 6) (Num 5))
evalS_t9 = evalS (Leq (Num 6) (Boolean False))
evalS_t10 = evalS (IsZero (Num 4))
evalS_t11 = evalS (IsZero (Boolean True))
evalS_t12 = evalS (If (Boolean True) (Num 4) (Num 5))
evalS_t13 = evalS (If (Boolean True) (Boolean True) (Num 5))
evalS_t14 = evalS (If (Boolean False) (Boolean True) (Num 5))



evalM :: Env -> BBAE -> (Maybe BBAE)
evalM _ (Num x) = Just (Num x)
evalM _ (Boolean x) = Just (Boolean x)
evalM e (Plus l r) = do {
  (Num l') <- evalM e l;
  (Num r') <- evalM e r;
  Just (Num (l' + r'))
}
evalM e (Minus l r) = do {
  (Num l') <- evalM e l;
  (Num r') <- evalM e r;
  if l' >= r' then return (Num (l' - r')) else Nothing
}
evalM e (Bind x a s) = do {
  a' <- evalM e a;
  (evalM ((x, a'):e) s)
}
evalM e (Id x) = do {
  v <- lookup x e;
  return v
}
evalM e (And l r) = do {
  (Boolean l') <- evalM e l;
  (Boolean r') <- evalM e r;
  return (Boolean (l' && r'))
}
evalM e (Leq l r) = do {
  (Num l') <- evalM e l;
  (Num r') <- evalM e r;
  if l' <=0 || r' <= 0 then Nothing else return (Boolean (l' <= r'))
}
evalM e (IsZero x) = do {
  (Num x') <- evalM e x;
  if x' < 0 then Nothing else return (Boolean (x' == 0))
}
evalM e (If a b c) = do {
  (Boolean a') <- evalS a;
  if a' then (evalM e b) else (evalM e c)
}

--evalM test cases
evalM_t1 = evalM [] (Num 5)
evalM_t2 = evalM [] (Plus (Num 5) (Num 6))
evalM_t3 = evalM [] (Minus (Num 6) (Num 5))
evalM_t4 = evalM [] (Minus (Num 5) (Num 6))
evalM_t5 = evalM [] (Boolean True)
evalM_t6 = evalM [] (Bind "x" (Num 3) (Plus (Num 4) (Id "x")))
evalM_t7 = evalM [] (Leq (Num 4) (Num 5))
evalM_t8 = evalM [] (Leq (Num 6) (Num 5))
evalM_t9 = evalM [] (Leq (Num 6) (Boolean False))
evalM_t10 = evalM [] (IsZero (Num 4))
evalM_t11 = evalM [] (IsZero (Boolean True))
evalM_t12 = evalM [] (If (Boolean True) (Num 4) (Num 5))
evalM_t13 = evalM [] (If (Boolean True) (Boolean True) (Num 5))
evalM_t14 = evalM [] (If (Boolean False) (Boolean True) (Num 5))

testBBAE :: BBAE -> Bool
testBBAE x = if evalS x == evalM [] x then True else False 

--testBBAE test cases
testBBAE_t1 = testBBAE (Num 5)
testBBAE_t2 = testBBAE (Boolean True)
testBBAE_t3 = testBBAE (Plus (Num 5) (Num 7))
testBBAE_t4 = testBBAE (Minus (Num 5) (Num 3))
testBBAE_t5 = testBBAE (Minus (Num 2) (Num 3))
testBBAE_t6 = testBBAE (Boolean True)
testBBAE_t7 = testBBAE (Bind "x" (Num 3) (Plus (Num 4) (Id "x")))
testBBAE_t8 = testBBAE (Leq (Num 4) (Num 5))
testBBAE_t9 = testBBAE (Leq (Num 6) (Num 5))
testBBAE_t10 = testBBAE (Leq (Num 6) (Boolean False))
testBBAE_t11 = testBBAE (IsZero (Num 4))
testBBAE_t12 = testBBAE (IsZero (Boolean True))
testBBAE_t13 = testBBAE (If (Boolean True) (Num 4) (Num 5))
testBBAE_t14 = testBBAE (If (Boolean True) (Boolean True) (Num 5))
testBBAE_t15 = testBBAE (If (Boolean False) (Boolean True) (Num 5))


typeofM :: Cont -> BBAE -> (Maybe TBBAE)
typeofM _ (Num _) = return TNum
typeofM _ (Boolean _) = return TBool
typeofM c (Plus l r) = do {
  TNum <- typeofM c l;
  TNum <- typeofM c r;
  return TNum
}
typeofM c (Minus l r) = do {
  TNum <- typeofM c l;
  TNum <- typeofM c r;
  return TNum
}
typeofM c (And l r) = do {
  TBool <- typeofM c l;
  TBool <- typeofM c r;
  return TBool
}
typeofM c (If x y z) = do {
  TBool <- typeofM c x;
  y' <- typeofM c y;
  z' <- typeofM c z;
  if y' == z' then return y' else Nothing
}
typeofM c (Bind x a s) = do {
  ta <- typeofM c a;
  typeofM ((x, ta):c) s
}
typeofM c (Id x) = lookup x c
typeofM c (Leq l r) = do {
  TNum <- typeofM c l;
  TNum <- typeofM c r;
  return TNum
}
typeofM c (IsZero x) = do {
  TNum <- typeofM c x;
  return TNum
}

--typeofM test cases
typeofM_t1 = typeofM [] (Num 5)
typeofM_t2 = typeofM [] (Boolean True)
typeofM_t3 = typeofM [] (Plus (Num 5) (Num 7))
typeofM_t4 = typeofM [] (Minus (Num 5) (Num 3))
typeofM_t5 = typeofM [] (Minus (Num 2) (Num 3))
typeofM_t6 = typeofM [] (Boolean True)
typeofM_t7 = typeofM [] (Bind "x" (Num 3) (Plus (Num 4) (Id "x")))
typeofM_t8 = typeofM [] (Leq (Num 4) (Num 5))
typeofM_t9 = typeofM [] (Leq (Num 6) (Num 5))
typeofM_t10 = typeofM [] (Leq (Num 6) (Boolean False))
typeofM_t11 = typeofM [] (IsZero (Num 4))
typeofM_t12 = typeofM [] (IsZero (Boolean True))
typeofM_t13 = typeofM [] (If (Boolean True) (Num 4) (Num 5))
typeofM_t14 = typeofM [] (If (Boolean True) (Boolean True) (Num 5))
typeofM_t15 = typeofM [] (If (Boolean False) (Boolean True) (Num 5))


evalT :: BBAE -> (Maybe BBAE)
evalT x = do {
  t' <- typeofM [] x;
  evalM [] x
}

--evalT test cases
evalT_t1 = evalT (Num 5)
evalT_t2 = evalT (Plus (Num 5) (Num 6))
evalT_t3 = evalT (Minus (Num 6) (Num 5))
evalT_t4 = evalT (Minus (Num 5) (Num 6))
evalT_t5 = evalT (Boolean True)
evalT_t6 = evalT (Bind "x" (Num 3) (Plus (Num 4) (Id "x")))
evalT_t7 = evalT (Leq (Num 4) (Num 5))
evalT_t8 = evalT (Leq (Num 6) (Num 5))
evalT_t9 = evalT (Leq (Num 6) (Boolean False))
evalT_t10 = evalT (IsZero (Num 4))
evalT_t11 = evalT (IsZero (Boolean True))
evalT_t12 = evalT (If (Boolean True) (Num 4) (Num 5))
evalT_t13 = evalT (If (Boolean True) (Boolean True) (Num 5))
evalT_t14 = evalT (If (Boolean False) (Boolean True) (Num 5))


run_test_cases = do
            putStrLn "Running test cases..."
            print "evalS test cases"
            print ("evalS (Num 5): ", evalS_t1)
            print ("evalS (Plus (Num 5) (Num 6)): ", evalS_t2)
            print ("evalS (Minus (Num 6) (Num 5)): ", evalS_t3)
            print ("evalS (Minus (Num 5) (Num 6)): ", evalS_t4)
            print ("evalS (Boolean True): ", evalS_t5)
            print ("evalS (Bind \"x\" (Num 3) (Plus (Num 4) (Id \"x\"))): ", evalS_t6)
            print ("evalS (Leq (Num 4) (Num 5)): ", evalS_t7)
            print ("evalS (Leq (Num 6) (Num 5)): ", evalS_t8)
            print ("evalS (Leq (Num 6) (Boolean False)): ", evalS_t9)
            print ("evalS (IsZero (Num 4)): ", evalS_t10)
            print ("evalS (IsZero (Boolean True)): ", evalS_t11)
            print ("evalS (If (Boolean True) (Num 4) (Num 5)): ", evalS_t12)
            print ("evalS (If (Boolean True) (Boolean True) (Num 5)): ", evalS_t13)
            print ("evalS (If (Boolean False) (Boolean True) (Num 5)): ", evalS_t14)

            print "evalM test cases"
            print ("evalM [] (Num 5): ", evalM_t1)
            print ("evalM [] (Plus (Num 5) (Num 6)): ", evalM_t2)
            print ("evalM [] (Minus (Num 6) (Num 5)): ", evalM_t3)
            print ("evalM [] (Minus (Num 5) (Num 6)): ", evalM_t4)
            print ("evalM [] (Boolean True): ", evalM_t5)
            print ("evalM [] (Bind \"x\" (Num 3) (Plus (Num 4) (Id \"x\"))): ", evalM_t6)
            print ("evalM [] (Leq (Num 4) (Num 5)): ", evalM_t7)
            print ("evalM [] (Leq (Num 6) (Num 5)): ", evalM_t8)
            print ("evalM [] (Leq (Num 6) (Boolean False)): ", evalM_t9)
            print ("evalM [] (IsZero (Num 4)): ", evalM_t10)
            print ("evalM [] (IsZero (Boolean True)): ", evalM_t11)
            print ("evalM [] (If (Boolean True) (Num 4) (Num 5)): ", evalM_t12)
            print ("evalM [] (If (Boolean True) (Boolean True) (Num 5)): ", evalM_t13)
            print ("evalM [] (If (Boolean False) (Boolean True) (Num 5)): ", evalM_t14)

            print "testBBAE test cases"
            print ("testBBAE (Num 5): ", testBBAE_t1)
            print ("testBBAE (Plus (Num 5) (Num 6)): ", testBBAE_t2)
            print ("testBBAE (Minus (Num 6) (Num 5)): ", testBBAE_t3)
            print ("testBBAE (Minus (Num 5) (Num 6)): ", testBBAE_t4)
            print ("testBBAE (Boolean True): ", testBBAE_t5)
            print ("testBBAE (Bind \"x\" (Num 3) (Plus (Num 4) (Id \"x\"))): ", testBBAE_t6)
            print ("testBBAE (Leq (Num 4) (Num 5)): ", testBBAE_t7)
            print ("testBBAE (Leq (Num 6) (Num 5)): ", testBBAE_t8)
            print ("testBBAE (Leq (Num 6) (Boolean False)): ", testBBAE_t9)
            print ("testBBAE (IsZero (Num 4)): ", testBBAE_t10)
            print ("testBBAE (IsZero (Boolean True)): ", testBBAE_t11)
            print ("testBBAE (If (Boolean True) (Num 4) (Num 5)): ", testBBAE_t12)
            print ("testBBAE (If (Boolean True) (Boolean True) (Num 5)): ", testBBAE_t13)
            print ("testBBAE (If (Boolean False) (Boolean True) (Num 5)): ", testBBAE_t14)

            print "typeofM test cases"
            print ("typeofM [] (Num 5): ", typeofM_t1)
            print ("typeofM [] (Plus (Num 5) (Num 6)): ", typeofM_t2)
            print ("typeofM [] (Minus (Num 6) (Num 5)): ", typeofM_t3)
            print ("typeofM [] (Minus (Num 5) (Num 6)): ", typeofM_t4)
            print ("typeofM [] (Boolean True): ", typeofM_t5)
            print ("typeofM [] (Bind \"x\" (Num 3) (Plus (Num 4) (Id \"x\"))): ", typeofM_t6)
            print ("typeofM [] (Leq (Num 4) (Num 5)): ", typeofM_t7)
            print ("typeofM [] (Leq (Num 6) (Num 5)): ", typeofM_t8)
            print ("typeofM [] (Leq (Num 6) (Boolean False)): ", typeofM_t9)
            print ("typeofM [] (IsZero (Num 4)): ", typeofM_t10)
            print ("typeofM [] (IsZero (Boolean True)): ", typeofM_t11)
            print ("typeofM [] (If (Boolean True) (Num 4) (Num 5)): ", typeofM_t12)
            print ("typeofM [] (If (Boolean True) (Boolean True) (Num 5)): ", typeofM_t13)
            print ("typeofM [] (If (Boolean False) (Boolean True) (Num 5)): ", typeofM_t14)

            print "evalT test cases"
            print ("evalT (Num 5): ", evalT_t1)
            print ("evalT (Plus (Num 5) (Num 6)): ", evalT_t2)
            print ("evalT (Minus (Num 6) (Num 5)): ", evalT_t3)
            print ("evalT (Minus (Num 5) (Num 6)): ", evalT_t4)
            print ("evalT (Boolean True): ", evalT_t5)
            print ("evalT (Bind \"x\" (Num 3) (Plus (Num 4) (Id \"x\"))): ", evalT_t6)
            print ("evalT (Leq (Num 4) (Num 5)): ", evalT_t7)
            print ("evalT (Leq (Num 6) (Num 5)): ", evalT_t8)
            print ("evalT (Leq (Num 6) (Boolean False)): ", evalT_t9)
            print ("evalT (IsZero (Num 4)): ", evalT_t10)
            print ("evalT (IsZero (Boolean True)): ", evalT_t11)
            print ("evalT (If (Boolean True) (Num 4) (Num 5)): ", evalT_t12)
            print ("evalT (If (Boolean True) (Boolean True) (Num 5)): ", evalT_t13)
            print ("evalT (If (Boolean False) (Boolean True) (Num 5)): ", evalT_t14)


main = do run_test_cases















