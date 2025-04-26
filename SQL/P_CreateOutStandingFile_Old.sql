if exists (select * from dbo.sysobjects where id = object_id(N'[dbo].[P_CreateOutStandingFile]') and OBJECTPROPERTY(id, N'IsProcedure') = 1)
drop procedure [dbo].[P_CreateOutStandingFile]
GO
CREATE	PROCEDURE	P_CreateOutStandingFile
AS
/*********************************************************************/
/*       WYETH CLOSURE – Create Separate Files                       */
/*********************************************************************/
/*     This is a BASIC but complete description of the purpose of    */
/*  this job and ONLY this job.  Do not explain the entire project.  */
/*  Also explain the steps in a broad, not detailed, format.         */
/*                                                                   */
/*                                                                   */
/*                                                                   */
/*                ---Date---  ---name---  --Purpose----------------- */
/*  Created       08/17/2009  EPAK                                   */
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
-- internal fields
/*********************************************************************/
DECLARE @LTIDOC		Char(6),
	@Process	Varchar(255),
	@Dest		Char(1),
	@SAP		Char(1),
	@GECS		Char(1),
	@Trust		Char(1),
	@Unknown	Char(1),
	@DC		Char(2),
	@CO		Char(2),
	@DO		Char(2),
	@ES		Char(2),
	@FS		Char(2),
	@UN		Char(2),
	@SAPFeed	Char(2),
	@Max		Int,
	@iMax		Int,
	@SeqNo		Int,
	@iSeqNo		Int,
	@One		Int,
	@Zero		Int,
	@OptNum		Int,
	@GrantNum	Int

/*********************************************************************/
-- STEP-01 -- collect parameters and pre-sets
/*********************************************************************/
SELECT	@step 		= '01--parameters and pre-sets'
SELECT	@FAIL		= 'Y'		-- Stays [Y] until end-of-job says all is OK...
SELECT	@start		= Getdate()
SELECT	@LTIDOC		= 'LTIDOC'
SELECT	@Process	= 'P_CreateOutStandingFile'
SELECT	@SAP		= 'S'
SELECT	@GECS		= 'A'
SELECT	@Trust		= 'T'
SELECT	@Unknown	= 'u'
SELECT	@DC		= 'DC'		-- Deferals with Cash
SELECT	@CO		= 'CO'		-- Cash Only
SELECT	@DO		= 'DO'		-- Deferal Only
SELECT	@ES		= 'ES'		-- Estate Only
SELECT	@FS		= 'FS'		-- FSE Only
SELECT	@UN		= 'UN'		-- Unknown
SELECT	@SAPFeed	= 'XX'
SELECT	@SeqNo		= 1
SELECT	@One		= 1
SELECT	@Zero		= 0

/*********************************************************************/
-- STEP-02 -- Create Temp Tables For Processing
/*********************************************************************/
SELECT	@step = '02a--Create #OPTNUM Temp Table'

Create	Table	#OPTNUM
 (
	SeqNo	Int	Identity,
	OptNum	Int	NOT NULL			
 )


SELECT 	@ERR = @@Error
IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

/*********************************************************************/
-- STEP-02 -- Create Temp Tables For Processing
/*********************************************************************/
SELECT	@step = '02b--Create #Determine Temp Table'

Create	Table	#Determine
 (
	SeqNo		Int	Identity,
	GrantNum	Int	NOT NULL,
	Destination	Char(1)	NOT NULL
 )

SELECT 	@ERR = @@Error
IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

/*********************************************************************/
-- STEP-03 -- SET Transaction
/*********************************************************************/
SELECT	@step = '03--SET Transaction'
BEGIN	TRAN	LTIDOCS		-- short-code of process-name

/*********************************************************************/
-- STEP-04 -- Determine FSE & Process
/*********************************************************************/
SELECT	@step = '04--Determain FSE'

Update	wyethaux.dbo.LTIDOC_OutstandingAwards
SET	SAPFeed	= @FS
From	wyethaux.dbo.LTIDOC_OutstandingAwards
Where	OPT_NUM IN
(
	Select	(OPT_NUM)
	From	Wyeth.dbo.Optionee
	Where	(RTRIM(USER_CD3) like '%fse%')
)
And	Destination IN (@SAP,@Trust)

SELECT 	@ERR = @@Error                   IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

/*********************************************************************/
-- STEP-05 -- Determine ESTATE & Process
/*********************************************************************/
SELECT	@step = '05--Determain Estates'

Update	wyethaux.dbo.LTIDOC_OutstandingAwards
SET	SAPFeed	= @ES
From	wyethaux.dbo.LTIDOC_OutstandingAwards
Where	OPT_NUM IN
(
	Select	(OPT_NUM)
	From	Wyeth.dbo.Optionee
	Where	(TERM_NUM = 6		--- Death
	Or	loc_cd = 'estate')
)
And	Destination IN (@SAP,@Trust)

SELECT 	@ERR = @@Error                   IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

/*********************************************************************/
-- STEP-06 -- Determine Deferal & Deferals with Cash & Cash Only
/*********************************************************************/
SELECT	@step = '06--Inset Into Temp Table #OPTNUM'

Insert	#OPTNUM
 (
	OptNum
 )
Select	DISTINCT OPT_NUM
From	wyethaux.dbo.LTIDOC_OutstandingAwards
Where	SAPFeed = @SAPFeed
And	Destination IN (@Trust,@SAP)

SELECT 	@ERR = @@Error                   IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

/*********************************************************************/
/*								     */
/*********************************************************************/
SELECT	@step = '06a--Inset Into Temp Table #OPTNUM Count'

Select	@Max	= Count(*)
From	#OPTNUM

SELECT 	@ERR = @@Error                   IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

/*********************************************************************/
/*								     */
/*********************************************************************/
WHILE @SeqNo <= @Max
 Begin	---- BEGIN OUTER LOOP
	/*************************************************************/
	/*							     */
	/*************************************************************/
	SELECT	@step = '10--Valuse @OptNum Variable'

	Select	@OptNum	= OptNum
	From	#OPTNUM
	Where	SeqNo = @SeqNo

	SELECT 	@ERR = @@Error                   	IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

	/*************************************************************/
	/*							     */
	/*************************************************************/
	SELECT	@step = '11--Truncate Internal LOOP Temp Table #Determine'

	Truncate Table	#Determine

	SELECT 	@ERR = @@Error                   	IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

	/*****************************************************************/
	/*						                 */
	/*****************************************************************/
	SET	@iMax		= @Zero
	SET	@iSeqNo		= @One

	/*****************************************************************/
	/*						                 */
	/*****************************************************************/	
	SELECT	@step = '12--Populate Internal Temp Table #Determine with Data'

	Insert	#Determine
	Select	Grant_Num,
		Destination
	From	wyethaux.dbo.LTIDOC_OutstandingAwards
	Where	OPT_NUM IN (@OptNum)
	And	Destination IN (@SAP, @Trust)

	SELECT 	@ERR = @@Error                   	IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

	/*****************************************************************/
	/*						                 */
	/*****************************************************************/
	Select	@iMax	= COUNT(*)
	From	#Determine

	/*****************************************************************/
	/*						                 */
	/*****************************************************************/
	WHILE @iSeqNo <= @iMax
	 Begin	---- BEGIN INNER LOOP
	 /****************************************************************/
	 /*					                 	 */
	 /****************************************************************/
	 Select	@GrantNum	= GrantNum,
		@Dest		= Destination
	 From	#Determine
	 Where	SeqNo 		= @iSeqNo

	 /****************************************************************/
	 /*					                 	 */
	 /****************************************************************/
	 IF (Select COUNT(DISTINCT Destination) From #Determine) = @One AND @Dest = @SAP
		Begin
		/*********************************************************/
		/*				                 	 */
		/*********************************************************/
		SELECT	@step = '20--Update LTIDOC_OutstandingAwards for Option Number: ' + RTRIM(CONVERT(Char,@OptNum))
 
		Update	wyethaux.dbo.LTIDOC_OutstandingAwards
		SET	SAPFeed		=  @CO
		Where	Opt_Num 	=  @OptNum
		And	Destination	IN (@Trust,@SAP)

		SELECT 	@ERR = @@Error                   		IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 
	
		/*********************************************************/
		/*				                 	 */
		/*********************************************************/
		End

	 /****************************************************************/
	 /*					                 	 */
	 /****************************************************************/
	 ELSE IF (Select COUNT(DISTINCT Destination) From #Determine) = @One AND @Dest = @Trust
		Begin
		/*********************************************************/
		/*				                 	 */
		/*********************************************************/
		SELECT	@step = '21--Update LTIDOC_OutstandingAwards for Option Number: ' + RTRIM(CONVERT(Char,@OptNum))

		Update	wyethaux.dbo.LTIDOC_OutstandingAwards
		SET	SAPFeed		= @DO
		Where	Opt_Num 	= @OptNum
		And	Destination	IN (@Trust,@SAP)

		SELECT 	@ERR = @@Error                   		IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

		/*********************************************************/
		/*				                 	 */
		/*********************************************************/
		End

	 /****************************************************************/
	 /*					                 	 */
	 /****************************************************************/
	 ELSE IF (Select COUNT(DISTINCT Destination) From #Determine) <> @One
		Begin
		/*********************************************************/
		/*				                 	 */
		/*********************************************************/
		SELECT	@step = '22--Update LTIDOC_OutstandingAwards for Option Number: ' + RTRIM(CONVERT(Char,@OptNum))

		Update	wyethaux.dbo.LTIDOC_OutstandingAwards
		SET	SAPFeed		= @DC
		Where	Opt_Num 	= @OptNum
		And	Destination	IN (@Trust,@SAP)

		SELECT 	@ERR = @@Error                   		IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

		/*********************************************************/
		/*				                 	 */
		/*********************************************************/
		End

	 /****************************************************************/
	 /*					                 	 */
	 /****************************************************************/
	 ELSE
		Begin
		/*********************************************************/
		/*				                 	 */
		/*********************************************************/
		SELECT	@step = '23--Update LTIDOC_OutstandingAwards for Option Number: ' + RTRIM(CONVERT(Char,@OptNum))

		Update	wyethaux.dbo.LTIDOC_OutstandingAwards
		SET	SAPFeed		= @UN
		Where	Opt_Num 	= @OptNum
		And	Destination	IN (@Trust,@SAP)

		SELECT 	@ERR = @@Error                   		IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

		/*********************************************************/
		/*				                 	 */
		/*********************************************************/
		End

	 /****************************************************************/
	 /*					                 	 */
	 /****************************************************************/
	 SELECT @iSeqNo = @iMax + @One

	 /****************************************************************/
	 /*					                 	 */
	 /****************************************************************/	
	 End	---- END INNER LOOP

	/*************************************************************/
	/*							     */
	/*************************************************************/
	Select	@SeqNo	= @SeqNo + @One

	/*************************************************************/
	/*							     */
	/*************************************************************/
 End	---- END OUTER LOOP

/*********************************************************************/
/*								     */
/*********************************************************************/
Drop	Table	#Determine
Drop	Table	#OPTNUM

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
	ROLLBACK TRAN	LTIDOCS
	INSERT	WYETHaux.dbo.LTIDOCProcessLog
	VALUES 	(Getdate(), @Process,@ERR,'FAILED',@Process + ' FAILED AT STEP : '+ @step)
	RAISERROR (51112,16,1)   -- Custom SQL error-message in MASTER
	RETURN @ERR
 End
ELSE
 Begin
	COMMIT	TRAN	LTIDOCS
	INSERT 	WYETHaux.dbo.LTIDOCProcessLog
	VALUES	(Getdate(), @Process,@ERR,'Success','Processed in ' + RTRIM(@elapse) + ' seconds.')
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