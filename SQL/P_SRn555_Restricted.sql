if exists (select * from dbo.sysobjects where id = object_id(N'[dbo].[P_SRn555_Restricted]') and OBJECTPROPERTY(id, N'IsProcedure') = 1)
drop procedure [dbo].[P_SRn555_Restricted]
GO
CREATE	PROCEDURE	P_SRn555_Restricted
AS
/*********************************************************************/
/*       WYETH CLOSURE – SRn55 Awardees                              */
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
/*  Created       06/17/2007  EPAK                                   */
/*  Mod 01          /  /       .                                     */
/*********************************************************************/
SET NOCOUNT ON	

/*********************************************************************/
-- STEP-01 -- DATA DECLARATIONS -- process log contents
/*********************************************************************/
DECLARE @start  	datetime	--DateTime of job start
DECLARE @end    	datetime	--DateTime of job end
DECLARE @DateOfRun	datetime	--Run Date/Time
DECLARE @elapse9 	decimal(6,2)	--Elapsed job-time in seconds
DECLARE @elapse 	char(6)		--Elapsed job-time (formatted)
DECLARE @LogDate	char(11)	--Holds the RunDate already written to the LOG

/*********************************************************************/
-- process control fields
/*********************************************************************/
DECLARE @ERR		integer		--Error-Handling : trap the return-code
DECLARE @FAIL		char(1)		--Error-Handling : was there an error?
DECLARE @step		varchar(30)	

/*********************************************************************/
-- static parameters
/*********************************************************************/
DECLARE	@CloseDateX	Varchar(8)	--Date of sale closure as text
DECLARE	@CloseDate	DateTime	--Date of sale closure as date
DECLARE	@ClosePriceX	Varchar(8)	--Price of sale closure as text
DECLARE	@ClosePrice	Money		--Price of sale closure as amount
DECLARE	@CashValueX	Varchar(8)	--Cash Value
DECLARE	@CashValue	Money		--Cash Value
DECLARE	@ShareValueX	Varchar(7)	--Cash Value of Pfizer Price
DECLARE	@ShareValue	Decimal(18,6)	--Cash Value of Pfizer Price
DECLARE	@CloseFactorX	Varchar(8)	--162M Factor
DECLARE	@CloseFactor	Decimal(18,6)	--162M Factor

/*********************************************************************/
-- internal fields
/*********************************************************************/
DECLARE @LTIDOC		Char(6),
	@Process	Varchar(8),
	@Counter	Int,
	@LTIDOCtotal	Int,
	@Dest		Char(1),
	@SAP		Char(1),
	@GECS		Char(1),
	@Trust		Char(1),
	@Unknown	Char(1)

/*********************************************************************/
-- STEP-02 -- collect parameters and pre-sets
/*********************************************************************/
SELECT	@step = '02--parameters and pre-sets'
SELECT	@FAIL		= 'Y'		-- Stays [Y] until end-of-job says all is OK...
SELECT	@start		= Getdate()
SELECT	@LTIDOC		= 'LTIDOC'
SELECT	@Process	= 'SRn555'
SELECT	@Dest		= 'z'
SELECT	@SAP		= 'S'
SELECT	@GECS		= 'A'
SELECT	@Trust		= 'T'
SELECT	@Unknown	= 'u'

/*--------------------------------------------------------------------*/
SELECT	@CloseDateX	=(	SELECT	PARAMvalue
				FROM	WYETHaux.dbo.tblParameters
				WHERE	RTRIM(PARAMproc) = 'LTIDOC'
				AND	RTRIM(PARAMname) = 'SaleDate'		)
SELECT 	@ERR = @@Error                   IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

SELECT	@CloseDate	= CONVERT(DateTime,@CloseDateX,108)
/*--------------------------------------------------------------------*/
SELECT	@ClosePriceX	= (	SELECT	PARAMvalue
				FROM	WYETHaux.dbo.tblParameters
				WHERE	RTRIM(PARAMproc) = 'LTIDOC'
				AND	RTRIM(PARAMname) = 'PfizerPrice'		)

SELECT 	@ERR = @@Error                   IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

SELECT	@ClosePrice	= Cast(@ClosePriceX as Money)
/*--------------------------------------------------------------------*/
SELECT	@CashValueX	= (	SELECT	PARAMvalue
				FROM	WYETHaux.dbo.tblParameters
				WHERE	RTRIM(PARAMproc) = 'LTIDOC'
				AND	RTRIM(PARAMname) = 'CashValue'		)

SELECT 	@ERR = @@Error                   IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

SELECT	@CashValue	= Cast(@CashValueX as Money)
/*--------------------------------------------------------------------*/
SELECT	@ShareValueX	= (	SELECT	PARAMvalue
				FROM	WYETHaux.dbo.tblParameters
				WHERE	RTRIM(PARAMproc) = 'LTIDOC'
				AND	RTRIM(PARAMname) = 'ShareValue'		)

SELECT 	@ERR = @@Error                   IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

SELECT	@ShareValue	= Cast(@ShareValueX as Decimal(18,6))
/*--------------------------------------------------------------------*/
SELECT	@CloseFactorX	= (	SELECT	PARAMvalue
				FROM	WYETHaux.dbo.tblParameters
				WHERE	RTRIM(PARAMproc) = 'LTIDOC'
				AND	RTRIM(PARAMname) = 'SaleFactor'		)

SELECT 	@ERR = @@Error                   IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

SELECT	@CloseFactor	= Cast(@CloseFactorX as Decimal(18,6)) / 100
/*--------------------------------------------------------------------*/

/*********************************************************************/
-- STEP-03 -- SET Transaction
/*********************************************************************/
SELECT	@step = '03--SET Transaction'
BEGIN	TRAN	LTIDOCSRn55	-- short-code of process-name

/*********************************************************************/
-- STEP-04 -- Cancel Grants from wyethaux.dbo.tblGrantzRestricted
/*********************************************************************/
SELECT	@step = '04--Cancel Grants from wyethaux.dbo.tblGrantzRestricted'

UPDATE	wyethaux.dbo.tblGrantzRestricted
SET	CANCEL1_SHARES	= LTI.oustanding_shares,
	CANCEL_DT	= @CloseDate,
	[USER_ID]	= @LTIDOC,
	ACTIV_DT	= Getdate()
From	wyethaux.dbo.LTIDOC_OutstandingAwards LTI INNER JOIN wyethaux.dbo.tblGrantzRestricted GR
ON	LTI.GRANT_NUM = GR.GRANT_NUM
Where	LTI.Process_id = @Process

SELECT 	@ERR = @@Error,
	@Counter = @@ROWCOUNTIF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

/*********************************************************************/
-- STEP-05 -- SET Transaction
SELECT	@step = '05--Update wyethaux.dbo.LTIDOC_OutstandingAwards'

UPDATE	wyethaux.dbo.LTIDOC_OutstandingAwards
SET	dist_dollars =	(((@ShareValue * @ClosePrice) * oustanding_shares) + (@CashValue * oustanding_shares)),
	processed_date = Getdate()
From	wyethaux.dbo.LTIDOC_OutstandingAwards
Where	Process_id = @Process

SELECT 	@ERR = @@Error                   IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

/*********************************************************************/
-- STEP-06 -- SET Transaction
SELECT	@step = '06--Total Outstanding Shares'

Select	@LTIDOCtotal = SUM(oustanding_shares)
From	wyethaux.dbo.LTIDOC_OutstandingAwards
Where	Process_id = @Process

SELECT 	@ERR = @@Error                   IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

/*********************************************************************/
-- STEP-07 -- Determine SAP or GECS or Override
SELECT	@step = '07a--Standard RSUs not 55/5 for GECS'

UPDATE	wyethaux.dbo.LTIDOC_OutstandingAwards
SET	destination = @GECS
From	wyethaux.dbo.LTIDOC_OutstandingAwards LTI INNER JOIN Wyeth.dbo.optionee OPT
ON	LTI.opt_num = OPT.opt_num
Where	LTI.Process_id = @Process
And	RTRIM(OPT.Address6) IN (SELECT RTRIM(GLOBAL_ID) FROM WYETHaux.dbo.tblGECSemployee)

SELECT 	@ERR = @@Error                   IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

/*********************************************************************/
SELECT	@step = '07b--Standard RSUs not 55/5 for SAP'

UPDATE	wyethaux.dbo.LTIDOC_OutstandingAwards
SET	destination = @SAP
From	wyethaux.dbo.LTIDOC_OutstandingAwards LTI INNER JOIN Wyeth.dbo.optionee OPT
ON	LTI.opt_num = OPT.opt_num
Where	LTI.Process_id = @Process
And	SUBSTRING(OPT.Address6,4,9) IN (SELECT GlobalID FROM WYETHaux.dbo.tblSAPimportUSA)

SELECT 	@ERR = @@Error                   IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

/*********************************************************************/
SELECT	@step = '07c--Standard RSUs not 55/5 for Override'

UPDATE	wyethaux.dbo.LTIDOC_OutstandingAwards
SET	destination = OVR.SAP_GECS
From	wyethaux.dbo.LTIDOC_OutstandingAwards LTI INNER JOIN wyethaux.dbo.LTIDOC_SAP_GECS_Override OVR
ON	OVR.OPT_NUM = LTI.OPT_NUM
Where	Process_id = @Process

SELECT 	@ERR = @@Error                   IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

/*********************************************************************/
-- STEP-08 -- Unknown
SELECT	@step = '08-- Can Not Determine - Standard RSUs not 55/5 for Unknown'

UPDATE	wyethaux.dbo.LTIDOC_OutstandingAwards
SET	destination = @Unknown
Where	Process_id = @Process
And	destination = @Dest

SELECT 	@ERR = @@Error                   IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

/*********************************************************************/
-- STEP-99 -- Completions and Closures			(Housekeeping and Logging)
/*********************************************************************/
SELECT	@step = '99--Completions and Closures'

SET 	@end      = getdate()
SET 	@elapse9  = (datediff(second,@start,@end))
SELECT 	@ERR = @@Error                   				
IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> -->

/*********************************************************************/
SELECT 	@elapse   = CAST(@elapse9 as varchar(8))

/*********************************************************************/
SELECT	@FAIL		= 'N'              -- No errors thus far?  Set error-flag to NO...

/*********************************************************************/
STEP99:	--	<-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- 
IF	@FAIL = 'Y'
 Begin
	ROLLBACK TRAN	LTIDOCSRn55
	INSERT	WYETHaux.dbo.LTIDOCProcessLog
	VALUES 	(Getdate(), 'Close-' + @Process,3,'FAILED',@Process + ' FAILED AT STEP : '+ @step)
	RAISERROR (51105,16,1)   -- Custom SQL error-message in MASTER
	RETURN 16
 End
ELSE
 Begin
	COMMIT	TRAN	LTIDOCSRn55

	INSERT 	WYETHaux.dbo.LTIDOCProcessLog
	VALUES	(Getdate(), 'Close-' + @Process,0,'Success','Processed ' +  ISNULL(wyethaux.dbo.fnAddCommaToNumber (@Counter),0) + ' awards for ' + ISNULL(wyethaux.dbo.fnAddCommaToNumber (@LTIDOCtotal),0) + ' stock shares in ' + RTRIM(@elapse) + ' seconds.')
 End

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
SET NOCOUNT OFF