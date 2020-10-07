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

--evalS _ = Nothing

evalM :: Env -> BBAE -> (Maybe BBAE)
evalM _ _ = Nothing

testBBAE :: BBAE -> Bool
testBBAE _ = True

typeofM :: Cont -> BBAE -> (Maybe TBBAE)
typeofM _ _ = Nothing

evalT :: BBAE -> (Maybe BBAE)
evalT _ = Nothing

