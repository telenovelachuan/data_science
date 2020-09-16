{-# LANGUAGE GADTs, FlexibleContexts #-}

-- Imports for Parsec
import Control.Monad
import Text.ParserCombinators.Parsec
import Text.ParserCombinators.Parsec.Language
import Text.ParserCombinators.Parsec.Expr
import Text.ParserCombinators.Parsec.Token

--
-- Simple caculator over naturals with no identifiers
--
-- Author: Perry Alexander
-- Date: Tue Jan 23 17:54:44 CST 2018
--
-- Source files for the Arithmetic Expressions (AE) language from PLIH
--

-- AST Definition

data AE where
  Num :: Int -> AE
  Plus :: AE -> AE -> AE
  Minus :: AE -> AE -> AE
  Mult :: AE -> AE -> AE
  Div :: AE -> AE -> AE
  If0 :: AE -> AE -> AE -> AE
  deriving (Show,Eq)

-- AE Parser (Requires ParserUtils and Parsec included above)

languageDef =
  javaStyle { identStart = letter
            , identLetter = alphaNum
            , reservedNames = [ "if0"
                              , "then"
                              , "else"
                              ]
            , reservedOpNames = [ "+","-","*","/"]
            }
  
lexer = makeTokenParser languageDef

inFix o c a = (Infix (reservedOp lexer o >> return c) a)
preFix o c = (Prefix (reservedOp lexer o >> return c))
postFix o c = (Postfix (reservedOp lexer o >> return c))

parseString p str =
  case parse p "" str of
    Left e -> error $ show e
    Right r -> r

expr :: Parser AE
expr = buildExpressionParser operators term

operators = [
  [ inFix "*" Mult AssocLeft
    , inFix "/" Div AssocLeft ]
  , [ inFix "+" Plus AssocLeft
  , inFix "-" Minus AssocLeft ]
  ]
  
numExpr :: Parser AE
numExpr = do i <- integer lexer
             return (Num (fromInteger i))

ifExpr :: Parser AE
ifExpr  = do reserved lexer "if0"
             c <- expr
             reserved lexer "then"
             t <- expr
             reserved lexer "else"
             e <- expr
             return (If0 c t e)
                     

term = parens lexer expr
       <|> numExpr
       <|> ifExpr

-- Parser invocation
-- Call parseAE to parse a string into the AE data structure.

parseAE = parseString expr


-- Evaluation Functions
-- Replace the bodies of these functions with your implementations for
-- Exercises 1-4.  Feel free to add utility functions or testing functions as
-- you see fit, but do not change the function signatures.  Note that only
-- Exercise 4 requires you to integrate the parser above.


evalAE :: AE -> Int
evalAE (Num x) = if x < 0 then error $ show "Input error! Please check your input number." else x
evalAE (Plus l r) = if (evalAE l) < 0 || (evalAE r) < 0 then error $ show "Input error!" else (evalAE l) + (evalAE r)
evalAE (Minus l r) = if (evalAE l) < 0 || (evalAE r) < 0 || ((evalAE l) < (evalAE r)) then error $ show "Input error!" else (evalAE l) - (evalAE r)
evalAE (Mult l r) = if (evalAE l) < 0 || (evalAE r) < 0 then error $ show "Input error!" else (evalAE l) * (evalAE r)
evalAE (Div l r) = if (evalAE l) < 0 || (evalAE r) <= 0 then error $ show "Input error!" else (evalAE l) `div` (evalAE r)
evalAE (If0 x y z) = case (evalAE x) of 
    0 -> case (evalAE y) of
              0 -> (evalAE z)
              _ -> (evalAE y)
    _ -> (evalAE x)


--evalAE test cases
pos1 = Num(10)
pos2 = Num(2)
pos3 = Num(4)
neg1 = Num(-3)
zero = Num(0)
evalAE_t1 = evalAE (pos1)
evalAE_t2 = evalAE (neg1)
evalAE_t3 = evalAE (Plus pos1 pos2)
evalAE_t4 = evalAE (Plus pos1 neg1)
evalAE_t5 = evalAE (Minus pos1 pos2)
evalAE_t6 = evalAE (Minus pos2 pos1)
evalAE_t7 = evalAE (Minus pos1 neg1)
evalAE_t8 = evalAE (Mult pos1 pos2)
evalAE_t9 = evalAE (Mult pos1 neg1)
evalAE_t10 = evalAE (Mult pos2 zero)
evalAE_t11 = evalAE (Div pos1 pos2)
evalAE_t12 = evalAE (Div pos1 neg1)
evalAE_t13 = evalAE (Div pos1 zero)
evalAE_t14 = evalAE (If0 pos1 pos2 pos3)
evalAE_t15 = evalAE (If0 zero pos2 pos3)
evalAE_t16 = evalAE (If0 zero neg1 pos3)
evalAE_t17 = evalAE (If0 zero zero pos3)
evalAE_t18 = evalAE (If0 zero zero zero)


evalAEMaybe :: AE -> Maybe Int
evalAEMaybe (Num x) = if x < 0 then Nothing else Just x
evalAEMaybe (Plus l r) = case (evalAEMaybe l) of
      Nothing -> Nothing
      Just l -> case (evalAEMaybe r) of
                  Nothing -> Nothing
                  Just r -> Just (l + r)
evalAEMaybe (Minus l r) = case (evalAEMaybe l) of
      Nothing -> Nothing
      Just l -> case (evalAEMaybe r) of
                  Nothing -> Nothing
                  Just r -> if l > r then Just (l - r) else Nothing
evalAEMaybe (Mult l r) = case (evalAEMaybe l) of
      Nothing -> Nothing
      Just l -> case (evalAEMaybe r) of
                  Nothing -> Nothing
                  Just r -> Just (l * r)
evalAEMaybe (Div l r) = case (evalAEMaybe l) of
      Nothing -> Nothing
      Just l -> case (evalAEMaybe r) of
                  Nothing -> Nothing
                  Just 0 -> Nothing
                  Just r -> Just (l `div` r)
evalAEMaybe (If0 x y z) = case (evalAEMaybe x) of 
    Just 0 -> case (evalAEMaybe y) of
              Just 0 -> (evalAEMaybe z)
              _ -> (evalAEMaybe y)
    _ -> (evalAEMaybe x)

--evalAEMaybe test cases
evalAEMaybe_t1 = evalAEMaybe (pos1)
evalAEMaybe_t2 = evalAEMaybe (neg1)
evalAEMaybe_t3 = evalAEMaybe (Plus pos1 pos2)
evalAEMaybe_t4 = evalAEMaybe (Plus pos1 neg1)
evalAEMaybe_t5 = evalAEMaybe (Minus pos1 pos2)
evalAEMaybe_t6 = evalAEMaybe (Minus pos2 pos1)
evalAEMaybe_t7 = evalAEMaybe (Minus pos1 neg1)
evalAEMaybe_t8 = evalAEMaybe (Mult pos1 pos2)
evalAEMaybe_t9 = evalAEMaybe (Mult pos1 neg1)
evalAEMaybe_t10 = evalAEMaybe (Mult pos2 zero)
evalAEMaybe_t11 = evalAEMaybe (Div pos1 pos2)
evalAEMaybe_t12 = evalAEMaybe (Div pos1 neg1)
evalAEMaybe_t13 = evalAEMaybe (Div pos1 zero)
evalAEMaybe_t14 = evalAEMaybe (If0 pos1 pos2 pos3)
evalAEMaybe_t15 = evalAEMaybe (If0 zero pos2 pos3)
evalAEMaybe_t16 = evalAEMaybe (If0 zero neg1 pos3)
evalAEMaybe_t17 = evalAEMaybe (If0 zero zero pos3)
evalAEMaybe_t18 = evalAEMaybe (If0 zero zero zero)

--liftNum :: Int -> Int -> Int
liftNum f num1 num2 = f num1 num2 

evalM :: AE -> Maybe Int
evalM (Num x) = if x < 0 then Nothing else return x
evalM (Plus l r) = do {l' <- (evalM l);
                       r' <- (evalM r);
                       Just (liftNum (+) l' r')}
evalM (Minus l r) = do {l' <- (evalM l);
                       r' <- (evalM r);
                       if l' >= r' then return (l' - r') else Nothing}
evalM (Mult l r) = do {l' <- (evalM l);
                       r' <- (evalM r);
                       Just (liftNum (*) l' r')}
evalM (Div l r) = do {l' <- (evalM l);
                      r' <- (evalM r);
                      if r' == 0 then Nothing else return (l' `div` r')}  
evalM (If0 x y z) = do {x' <- (evalM x);
                        y' <- (evalM y);
                        z' <- (evalM z);
                        case x' of
                          0 -> case y' of 
                                      0 -> Just z'
                                      _ -> Just y'
                          _ -> Just x'} 
                   
--evalM _ = Nothing

--evalM test cases
evalM_t1 = evalM (pos1)
evalM_t2 = evalM (neg1)
evalM_t3 = evalM (Plus pos1 pos2)
evalM_t4 = evalM (Plus pos1 neg1)
evalM_t5 = evalM (Minus pos1 pos2)
evalM_t6 = evalM (Minus pos2 pos1)
evalM_t7 = evalM (Minus pos1 neg1)
evalM_t8 = evalM (Mult pos1 pos2)
evalM_t9 = evalM (Mult pos1 neg1)
evalM_t10 = evalM (Mult pos2 zero)
evalM_t11 = evalM (Div pos1 pos2)
evalM_t12 = evalM (Div pos1 neg1)
evalM_t13 = evalM (Div pos1 zero)
evalM_t14 = evalM (If0 pos1 pos2 pos3)
evalM_t15 = evalM (If0 zero pos2 pos3)
evalM_t16 = evalM (If0 zero neg1 pos3)
evalM_t17 = evalM (If0 zero zero pos3)
evalM_t18 = evalM (If0 zero zero zero)

interpAE :: String -> Maybe Int
interpAE str = evalM (parseAE str)
--interpAE _ = Nothing

--interpAE test cases
interpAEt1 = interpAE "1+3"
interpAEt2 = interpAE "(1+3)*5 + 1"
interpAEt3 = interpAE "(1+3)*5/0"
interpAEt4 = interpAE "(1+3)*(-1)/8"
interpAEt5 = interpAE "(1+3)*2-100"


run_test_cases = do
            putStrLn "Running test cases..."
            print "evalAE test cases"
            print ("evalAE (Num 10): ", evalAE_t1)
            --print ("evalAE (Num -3): ", evalAE_t2)
            print ("evalAE Plus ((Num 10) (Num 2)): ", evalAE_t3)
            --print ("evalAE Plus ((Num 10) (Num -3)): ", evalAE_t4)
            print ("evalAE Minus ((Num 10) (Num 2)): ", evalAE_t5)
            --print ("evalAE Minus ((Num 2) (Num 10)): ", evalAE_t6)
            --print ("evalAE Minus ((Num 10) (Num -3)): ", evalAE_t7)
            print ("evalAE Mult ((Num 10) (Num 2)): ", evalAE_t8)
            --print ("evalAE Mult ((Num 10) (Num -3)): ", evalAE_t9)
            print ("evalAE Mult ((Num 10) (Num 0)): ", evalAE_t10)
            print ("evalAE Div ((Num 10) (Num 2)): ", evalAE_t11)
            --print ("evalAE Div ((Num 10) (Num -3)): ", evalAE_t12)
            --print ("evalAE Div ((Num 10) (Num 0)): ", evalAE_t13)
            print ("evalAE If0 ((Num 10) (Num 2) (Num 4)): ", evalAE_t14)
            print ("evalAE If0 ((Num 0) (Num 2) (Num 4)): ", evalAE_t15)
            --print ("evalAE If0 ((Num 0) (Num -3) (Num 4)): ", evalAE_t16)
            print ("evalAE If0 ((Num 0) (Num 0) (Num 4)): ", evalAE_t17)
            print ("evalAE If0 ((Num 0) (Num 0) (Num 0)): ", evalAE_t18)


            print "evalAEMaybe test cases"
            print ("evalAEMaybe (Num 10): ", evalAEMaybe_t1)
            print ("evalAEMaybe (Num -3): ", evalAEMaybe_t2)
            print ("evalAEMaybe Plus ((Num 10) (Num 2)): ", evalAEMaybe_t3)
            print ("evalAEMaybe Plus ((Num 10) (Num -3)): ", evalAEMaybe_t4)
            print ("evalAEMaybe Minus ((Num 10) (Num 2)): ", evalAEMaybe_t5)
            print ("evalAEMaybe Minus ((Num 2) (Num 10)): ", evalAEMaybe_t6)
            print ("evalAEMaybe Minus ((Num 10) (Num -3)): ", evalAEMaybe_t7)
            print ("evalAEMaybe Mult ((Num 10) (Num 2)): ", evalAEMaybe_t8)
            print ("evalAEMaybe Mult ((Num 10) (Num -3)): ", evalAEMaybe_t9)
            print ("evalAEMaybe Mult ((Num 10) (Num 0)): ", evalAEMaybe_t10)
            print ("evalAEMaybe Div ((Num 10) (Num 2)): ", evalAEMaybe_t11)
            print ("evalAEMaybe Div ((Num 10) (Num -3)): ", evalAEMaybe_t12)
            print ("evalAEMaybe Div ((Num 10) (Num 0)): ", evalAEMaybe_t13)
            print ("evalAEMaybe If0 ((Num 10) (Num 2) (Num 4)): ", evalAEMaybe_t14)
            print ("evalAEMaybe If0 ((Num 0) (Num 2) (Num 4)): ", evalAEMaybe_t15)
            print ("evalAEMaybe If0 ((Num 0) (Num -3) (Num 4)): ", evalAEMaybe_t16)
            print ("evalAEMaybe If0 ((Num 0) (Num 0) (Num 4)): ", evalAEMaybe_t17)
            print ("evalAEMaybe If0 ((Num 0) (Num 0) (Num 0)): ", evalAEMaybe_t18)

            print "evalM test cases"
            print ("evalM (Num 10): ", evalM_t1)
            print ("evalM (Num -3): ", evalM_t2)
            print ("evalM Plus ((Num 10) (Num 2)): ", evalM_t3)
            print ("evalM Plus ((Num 10) (Num -3)): ", evalM_t4)
            print ("evalM Minus ((Num 10) (Num 2)): ", evalM_t5)
            print ("evalM Minus ((Num 2) (Num 10)): ", evalM_t6)
            print ("evalM Minus ((Num 10) (Num -3)): ", evalM_t7)
            print ("evalM Mult ((Num 10) (Num 2)): ", evalM_t8)
            print ("evalM Mult ((Num 10) (Num -3)): ", evalM_t9)
            print ("evalM Mult ((Num 10) (Num 0)): ", evalM_t10)
            print ("evalM Div ((Num 10) (Num 2)): ", evalM_t11)
            print ("evalM Div ((Num 10) (Num -3)): ", evalM_t12)
            print ("evalM Div ((Num 10) (Num 0)): ", evalM_t13)
            print ("evalM If0 ((Num 10) (Num 2) (Num 4)): ", evalM_t14)
            print ("evalM If0 ((Num 0) (Num 2) (Num 4)): ", evalM_t15)
            print ("evalM If0 ((Num 0) (Num -3) (Num 4)): ", evalM_t16)
            print ("evalM If0 ((Num 0) (Num 0) (Num 4)): ", evalM_t17)
            print ("evalM If0 ((Num 0) (Num 0) (Num 0)): ", evalM_t18)

            print "interpAE test cases"
            print ("1+3: ", interpAEt1)
            print ("(1+3)*5 + 1: ", interpAEt2)
            print ("(1+3)*5/0: ", interpAEt3)
            print ("(1+3)*(-1)/8: ", interpAEt4)
            print ("(1+3)*2-100: ", interpAEt5)
            putStrLn "All test cases executed!"

main = do run_test_cases



