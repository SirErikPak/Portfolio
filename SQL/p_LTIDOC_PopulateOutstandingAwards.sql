if exists (select * from dbo.sysobjects where id = object_id(N'[dbo].[p_LTIDOC_PopulateOutstandingAwards]') and OBJECTPROPERTY(id, N'IsProcedure') = 1)
drop procedure [dbo].[p_LTIDOC_PopulateOutstandingAwards]
GO
CREATE	PROCEDURE	p_LTIDOC_PopulateOutstandingAwards
AS

/*********************************************************************/
/*       WYETH CLOSURE – p_LTIDOC_POATp_LTIDOC_POATp_LTIDOC_POATp_   */
/*	  LTIDOC_POATp_LTIDOC_POATp_LTIDOC_POATp_LTIDOC_POATxxx      */
/*********************************************************************/
/*     This is a BASIC but complete description of the purpose of    */
/*  this job and ONLY this job.  Do not explain the entire project.  */
/*  Also explain the steps in a broad, not detailed, format.         */
/*                                                                   */
/*  STEPS:                                                           */
/*      a)  Extract options that are underwater                      */
/*      b)  assemble cancellation records to new temp-file           */
/*      c)  assemble a grid report for users' reference              */
/*                                                                   */
/*                ---Date---  ---name---  --Purpose----------------- */
/*  Created       06/12/2009   C.Fowler                              */
/*  Mod 01          /  /       .                                     */	--MOD-01
/*********************************************************************/

SET	NOCOUNT		ON	

/***************************************************************************************/
-- STEP-01 -- DATA DECLARATIONS 
-- process log contents
DECLARE @start  	datetime	--DateTime of job start
DECLARE @end    	datetime	--DateTime of job end
DECLARE @DateOfRun	datetime	--Run Date/Time
DECLARE @elapse9 	decimal(6,2)	--Elapsed job-time in seconds

DECLARE @LogDate	char(011)	--Holds the RunDate already written to the LOG
-- process control fields
DECLARE @ERR		integer		--Error-Handling : trap the return-code
DECLARE @FAIL		char(01)	--Error-Handling : was there an error?
DECLARE @step		varchar(30)	
-- static parameters
DECLARE	@CloseDateX	Char(08)	--Date of sale closure as text
DECLARE	@CloseDate	DateTime	--Date of sale closure as date
DECLARE	@ClosePriceX	Char(08)	--Price of sale closure as text
DECLARE	@ClosePrice	Money		--Price of sale closure as amount
DECLARE	@CashValueX	Char(08)	--Fixed rate value for cash modification upon closure as text
DECLARE	@CashValue	Money		--Fixed rate value for cash modification upon closure as amount
DECLARE	@ShareValueX	Char(08)	--Share percentage of Pfizer stock at closure as text
DECLARE	@ShareValue	Money		--Share percentage of Pfizer stock at closure as amount
DECLARE	@CloseFactorX	Char(08)	--Modification percentage of 162M units at closure as text
DECLARE	@CloseFactor	Float		--Modification percentage of 162M units at closure as amount
-- internal fields
DECLARE	@AsOf		DateTime	-- all activity after is ignored
DECLARE	@After		DateTime	-- all activity before is ignored
DECLARE	@Consideration	Money		-- Set this parameter price of value on closing


/***************************************************************************************/
-- STEP-02 -- collect parameters and pre-sets
SELECT	@step = '02--parameters and pre-sets'
SELECT	@FAIL		= 'Y'		-- Stays [Y] until end-of-job says all is OK...
SELECT	@start		= getdate()
/*-------------------------------------------------------------------------------------*/
SELECT	@CloseDateX	=(	SELECT	PARAMvalue
				FROM	WYETHaux.dbo.tblParameters
				WHERE	PARAMproc = 'LTIDOC'
				AND	PARAMname = 'SaleDate'		)
SELECT 	@ERR = @@Error                   
IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

SELECT	@CloseDate	= CONVERT(DateTime,@CloseDateX,108)
/*-------------------------------------------------------------------------------------*/
SELECT	@ClosePriceX	= (	SELECT	PARAMvalue
				FROM	WYETHaux.dbo.tblParameters
				WHERE	PARAMproc = 'LTIDOC'
				AND	PARAMname = 'PfizerPrice'		)

SELECT 	@ERR = @@Error                   
IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

SELECT	@ClosePrice	= Cast(@ClosePriceX as Money)
/*-------------------------------------------------------------------------------------*/
SELECT	@CashValueX	= (	SELECT	PARAMvalue
				FROM	WYETHaux.dbo.tblParameters
				WHERE	PARAMproc = 'LTIDOC'
				AND	PARAMname = 'CashValue'		)

SELECT 	@ERR = @@Error                   
IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

SELECT	@CashValue	= Cast(@CashValueX as Money)
/*-------------------------------------------------------------------------------------*/
SELECT	@ShareValueX	= (	SELECT	PARAMvalue
				FROM	WYETHaux.dbo.tblParameters
				WHERE	PARAMproc = 'LTIDOC'
				AND	PARAMname = 'ShareValue'		)

SELECT 	@ERR = @@Error                   
IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

SELECT	@ShareValue	= Cast(@ShareValueX as Money)
/*-------------------------------------------------------------------------------------*/
SELECT	@CloseFactorX	= (	SELECT	PARAMvalue
				FROM	WYETHaux.dbo.tblParameters
				WHERE	PARAMproc = 'LTIDOC'
				AND	PARAMname = 'SaleFactor'		)

SELECT 	@ERR = @@Error                   
IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

SELECT	@CloseFactor	= Cast(@CloseFactorX as float) / 100

SET	@AsOf	= @CloseDateX		
SET	@After	= '9/1/1999'		
SET	@Consideration	= @ClosePrice + @CashValue	
/*-------------------------------------------------------------------------------------*/


-- STEP-03 -- Empty Outstanding Awards table first 
TRUNCATE TABLE wyethaux.dbo.LTIDOC_OutstandingAwards

 
-- STEP-04 -- Collect eligible people into TEMP
/*
SELECT	opt_num, sub_cd, birth_dt, hire_dt, opt_id

INTO	#optionees
FROM	wyeth.dbo.optionee 
*/

--STEP-05 -- Collect Option Grants
SELECT	grt.opt_num, grt.GRANT_NUM, grt.GRANT_DT, grt.OPT_PRC,
	-- Process Id for IMSO & UWSO is decided here!! 
	case when grt.opt_prc < @Consideration then 'IMSO    '
		else 'UWSO    ' end as process_id,
	grt.OPTS_GRNTD,
	(SELECT	SUM(IsNull(OPTS_EXER,0)) 
	 FROM	wyeth.dbo.exercise exe
	 WHERE	exe.GRANT_NUM = grt.GRANT_NUM
	 AND	exe.EXER_DT < @AsOf) 				as EXEshares,
	(SELECT	SUM(IsNull(OPTS_CANC,0)) 
	 FROM	wyeth.dbo.cancel can
	 WHERE	can.GRANT_NUM = grt.GRANT_NUM
	 AND	can.CANC_DT < @AsOf) 				as CANshares,
	(SELECT	SUM(GRTDETshares)
	 FROM	wyethaux.dbo.tblGrantTranches
	 WHERE	GRTDETgrant = grt.GRANT_NUM
	 AND	GRTDETdate <= @AsOf) 				as VSTshares,
	0							as VOUshares,
	0 							as OUTshares,
	CAST(0 as Money)					as NETvalue,
	@Consideration as per_share, pln.PLAN_ID, tgd.deferelectiontype as Defer_flag, grt.grant_cd6 as user_cd3
INTO	#grants
FROM		wyeth.dbo.grantz 	grt
JOIN		wyeth.dbo.planz 	pln ON pln.PLAN_NUM = grt.PLAN_NUM
--JOIN		#optionees 			opt ON opt.OPT_NUM = grt.OPT_NUM
JOIN		wyeth.dbo.optionee	opt ON opt.OPT_NUM = grt.OPT_NUM
LEFT JOIN	wyethaux.dbo.tblgrantdefferals 	tgd ON tgd.grantnumber = grt.grant_num 
WHERE	grt.GRANT_DT < @AsOf
--AND	grt.GRANT_DT >= @After
and opt.sub_cd <> '0101'   --- NED

 
--STEP-06 -- Collect Restricted Grants
INSERT	#grants
SELECT	grt.OPT_NUM, grt.GRANT_NUM, grt.GRANT_DT, grt.OPT_PRC,
	-- Process Id for 162M, ODRP, SRn555, PnD, DDRT is decided here!!
 
	CASE when grt.grant_cd6 like '%162%' then '162M'
  		when tgd.deferelectiontype is not null then 'ODRP'
  		when grt.plan_type in (3,4) and 
    			   ( ( DateDiff(day,opt.BIRTH_DT,'12/31/2009')/365.25   <   55.0
      			     OR DateDiff(day,opt.HIRE_DT ,'12/31/2009')/365.25   <   05.0 )
     			   and year(grt.grant_dt)  =  2007       )
   			or
    			   (  ( DateDiff(day,opt.BIRTH_DT,'12/31/2010')/365.25   <   55.0
     			     OR DateDiff(day,opt.HIRE_DT ,'12/31/2010')/365.25   <   05.0 )
     			   and year(grt.grant_dt)  =  2008       )
   			or
    			   opt.opt_id not like 'U%'
			or
			   (opt.opt_id like 'U%' and year(grt.grant_dt) not in (2007,2008)) 
   			then 'SRn555'
  		when grt.plan_type = 2 and (tgd.deferelectiontype is null) then 'PnD' 
  		when opt.opt_id like 'U%' then 'DDRT'
  		ELSE 'TBD' END,
	

	grt.OPTS_GRNTD,
	case when grt.plan_type = 2 then
		case when tdr.exer_dt <= @AsOf then grt.OPTS_GRNTD
			when grt.perf_year < 2009 then grt.OPTS_GRNTD
			else 0 end
	else 
		(CASE WHEN grt.VEST_DT < @AsOf then grt.OPTS_GRNTD ELSE 0 END)end,
	( CASE WHEN grt.CANCEL_DT < @AsOf then grt.CANCEL1_SHARES ELSE 0 END) 
	+(CASE WHEN grt.CANCEL2_DT < @AsOf then grt.CANCEL2_SHARES ELSE 0 END),
	
	(CASE WHEN grt.VEST_DT < @AsOf then grt.OPTS_GRNTD ELSE 0 END), 
--	CASE WHEN grt.VEST_DT < @AsOf then grt.OPTS_GRNTD - grt.CANCEL1_SHARES ELSE 0 END,
	0, 0, 0, @Consideration,
	pln.PLAN_ID, tgd.deferelectiontype, grt.grant_cd6
FROM		wyethaux.dbo.tblGrantzRestricted 	grt
JOIN		wyeth.dbo.planz 			pln ON pln.PLAN_NUM = grt.PLAN_NUM
--JOIN		#optionees 					opt ON opt.OPT_NUM = grt.OPT_NUM
JOIN		wyeth.dbo.optionee				opt ON opt.OPT_NUM = grt.OPT_NUM
LEFT JOIN	wyethaux.dbo.tblgrantdefferals 			tgd ON tgd.grantnumber = grt.grant_num
left join	wyethaux.dbo.tbldistributionrestricted		tdr on grt.grant_num = tdr.grant_num 
WHERE	grt.GRANT_DT < @AsOf
---AND	grt.GRANT_DT >= @After
and 	opt.sub_cd <> '0101'  --- NEDR

-- STEP-89 -- Repair nulls in extracts
UPDATE	#grants
SET	EXEshares = IsNull(EXEshares,0),
	CANshares = IsNull(CANshares,0),
	VSTshares = IsNull(VSTshares,0),
--	VOUshares = IsNull(VOUshares,0),
	VOUshares = IsNull(VSTshares,0) - (IsNull(EXEshares,0) + IsNull(CANshares,0) ),
	OUTshares = OPTS_GRNTD - (IsNull(EXEshares,0) + IsNull(CANshares,0) ),
	NETvalue  = 0, 
--		case PLAN_TYPE WHEN  0 THEN 
--		   		(OPTS_GRNTD - (IsNull(EXEshares,0) + IsNull(CANshares,0) )) * (@Consideration - OPT_PRC)
--				WHEN 1 THEN 
--		   		(OPTS_GRNTD - (IsNull(EXEshares,0) + IsNull(CANshares,0) )) * (@Consideration - OPT_PRC)
--				ELSE 
--		   		(OPTS_GRNTD - (IsNull(EXEshares,0) + IsNull(CANshares,0) )) * @Consideration 
--				END,
	Defer_flag = isnull(Defer_flag,'')
 

-- STEP-90 -- Extract data into Outstinding Awards Table

select		'Before Update'
select 		* from wyethaux.dbo.LTIDOC_OutstandingAwards


insert 	wyethaux.dbo.LTIDOC_OutstandingAwards
(
	opt_num,grant_num ,process_id,oustanding_shares
)
--SELECT		opt.opt_num, grt.grant_num, rtrim(process_id), OUTshares, 0.00, 0,0, 'z'	07/31/2009
SELECT		opt.opt_num, grt.grant_num, rtrim(process_id), OUTshares
FROM		#grants grt
--JOIN		#optionees opt ON opt.OPT_NUM = grt.OPT_NUM
JOIN		wyeth.dbo.optionee opt ON opt.OPT_NUM = grt.OPT_NUM
WHERE		OUTshares > 0

-- Should balance to EOWin's report "Options_Outstanding"
select	'Opts-Outstanding'
SELECT	SUM(oustanding_shares)
FROM	wyethaux.dbo.LTIDOC_OutstandingAwards
WHERE	process_ID IN ('IMSO','UWSO')

select		'After Update'
select 		* from wyethaux.dbo.LTIDOC_OutstandingAwards
ORDER	BY	process_id, opt_num, grant_num


/*********************************************************************/
/*             FULL DESCRIPTION of the BUSINESS RULES                */
/*********************************************************************/
/*                                                                   */
/*  HERE is where detailed descriptions of filters, choices and      */
/*  will go.                                                         */
/*                                                                   */
/*  HERE is where destination of any built tables will be exported   */
/*  and sent to.                                                     */
/*                                                                   */
/*                                                                   */
/*********************************************************************/



/***************************************************************************************/
-- STEP-99 -- Completions and Closures			(Housekeeping and Logging)
DECLARE @elapse 	char(06)	--Elapsed job-time (formatted)
SELECT	@step = '99--Completions and Closures'

SET 	@end      = getdate()
SELECT 	@elapse9  = (datediff(second,@start,@end))
SELECT 	@ERR = @@Error                   				
IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> -->

SELECT 	@elapse   = CAST(@elapse9 as varchar(8))
/*-------------------------------------------------------------------------------------*/
SELECT	@FAIL		= 'N'              -- No errors thus far?  Set error-flag to NO...

STEP99:	--	<-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- 
IF	@FAIL = 'Y'
Begin
	--ROLLBACK	TRAN	p_LTIDOC_POAT
	INSERT 	INTO 		WYETHaux.dbo.LTIDOCProcessLog
		VALUES 	(Getdate(), 'p_LTIDOC_POAT',3,'FAILED','p_LTIDOC_POAT FAILED AT STEP : '+ @step)
	RAISERROR (51101,16,1)   -- Custom SQL error-message in MASTER
	RETURN 16
End
ELSE
Begin
	--COMMIT	TRAN	p_LTIDOC_POAT
--	INSERT 	INTO 	WYETHaux.dbo.LTIDOCProcessLog
--		VALUES 	(Getdate(), 'Close-p_LTIDOC_POAT',0,'Success','Created ' + 0 + ' cancels for ' 
--			+ 0 + ' stock shares')
	INSERT	INTO	WYETHaux.dbo.LTIDOCProcessLog
		VALUES	(Getdate(), 'p_LTIDOC_POAT',0,'Success','Done in ' + @elapse + ' seconds.')
End

--DROP	TABLE	#optionees
DROP	TABLE	#grants


select	*
from	wyethaux.dbo.LTIDOCprocesslog
order	by	date_time_stamp		desc 
 
SET	NOCOUNT	OFF
