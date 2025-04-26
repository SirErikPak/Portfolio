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
	@OptNum		Int,
	@GrantNum	Int,
	@RecordCount	Int,
	@Yes		Char(1),
	@No		Char(1),
	@TType		Char(1),
	@RType		Char(1),
	@InfoType	Char(4),
	@USD		Char(3),
	@GID		Char(3),
	@Tick		Char(1),
	@Tran		Char(1),
	@SapCMD		Varchar(8000),
	@Command	Varchar(8000),
	@TableName	Varchar(255),
	@Filename	Varchar(500),
	@FiileDest	Varchar(500),
	@ErrorName	Varchar(500),
	@LogName	Varchar(500)

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
SELECT	@iSeqNo		= 1
SELECT	@One		= 1
SELECT	@Yes		= 'Y'
SELECT	@No		= 'N'
SELECT	@TType		= 'T'
SELECT	@RType		= 'D'
SELECT	@InfoType	= '0015'
SELECT	@USD		= 'USD'
SELECT	@GID		= 'GID'
SELECT	@Tran		= 'Y'
SELECT	@Tick		= ''''

/*********************************************************************/
-- STEP-02 -- Create Temp Tables For Processing
/*********************************************************************/
SELECT	@step = '02a--Create #OPTNUM Temp Table'

Create	Table	#OPTNUM
 (
	SeqNo			Int		Identity,
	OptNum			Int		NOT NULL
 )


SELECT 	@ERR = @@Error
IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

/*********************************************************************/
-- STEP-02 -- Create Temp Tables For Processing
/*********************************************************************/
SELECT	@step = '02b--Create #SAPFeed Temp Table'

Create	Table	#SAPFeed
 (
	SeqNo		Int		Identity,
	SAPFeed		Char(2)		NOT NULL,
	TableName	Varchar(255)	NOT NULL,
	[FileName]	Varchar(255)	NOT NULL,
	FileLoc		Varchar(255)	NOT NULL,
	FileDest	Varchar(255)	NOT NULL,
	FileLogName	Varchar(255)	NOT NULL,
	FileErrorName	Varchar(255)	NOT NULL,
	FileLog		Varchar(255)	NOT NULL
 )

SELECT 	@ERR = @@Error
IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

/*********************************************************************/
/*     		Begin Transaction				     */
/*********************************************************************/
BEGIN	TRAN	LTIDOCS		-- short-code of process-name

/*********************************************************************/
-- STEP-03 -- SET Transaction
/*********************************************************************/
SELECT	@step = '03--Truncate  LTIDOCTempEndOfWyethAward Table'

Truncate	Table	LTIDOCTempEndOfWyethAward

SELECT 	@ERR = @@Error
IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

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
Order By OPT_NUM

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
	Select	@OptNum		= OptNum
	From	#OPTNUM
	Where	SeqNo 		= @SeqNo

	/*************************************************************/
	/*							     */
	/*************************************************************/
	SELECT	@step = '20--Update LTIDOC_OutstandingAwards for Option Number: ' + RTRIM(CONVERT(Char,@OptNum))

	/*************************************************************/
	/*							     */
	/*************************************************************/
	IF (Select COUNT(DISTINCT destination) From wyethaux.dbo.LTIDOC_OutstandingAwards 
		Where	SAPFeed = @SAPFeed And Destination IN (@Trust,@SAP)And Opt_Num = @OptNum) = @One 

	 Begin
		/*********************************************************/
		/*				                 	 */
		/*********************************************************/
		IF (Select DISTINCT destination From wyethaux.dbo.LTIDOC_OutstandingAwards 
			Where	SAPFeed = @SAPFeed And Destination IN (@Trust,@SAP)And Opt_Num = @OptNum) = @SAP
		 Begin
			/*************************************************/
			/*			                 	 */
			/*************************************************/
			Update	wyethaux.dbo.LTIDOC_OutstandingAwards
			SET	SAPFeed		=  @CO
			Where	Opt_Num 	=  @OptNum
			And	Destination	IN (@Trust,@SAP)

			SELECT 	@ERR = @@Error                   			IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

			/*************************************************/
			/*			                 	 */
			/*************************************************/
		 End
 
		/*********************************************************/
		/*				                 	 */
		/*********************************************************/
		IF (Select DISTINCT destination From wyethaux.dbo.LTIDOC_OutstandingAwards 
			Where	SAPFeed = @SAPFeed And Destination IN (@Trust,@SAP)And Opt_Num = @OptNum) = @Trust
		 Begin
			/*************************************************/
			/*			                 	 */
			/*************************************************/
			Update	wyethaux.dbo.LTIDOC_OutstandingAwards
			SET	SAPFeed		=  @DO
			Where	Opt_Num 	=  @OptNum
			And	Destination	IN (@Trust,@SAP)

			SELECT 	@ERR = @@Error                   			IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

			/*************************************************/
			/*			                 	 */
			/*************************************************/
		 End

	
		/*********************************************************/
		/*				                 	 */
		/*********************************************************/
	 End

	 /****************************************************************/
	 /*					                 	 */
	 /****************************************************************/
	 ELSE IF (Select COUNT(DISTINCT destination) From wyethaux.dbo.LTIDOC_OutstandingAwards 
			Where	SAPFeed = @SAPFeed And Destination IN (@Trust,@SAP)And Opt_Num = @OptNum) <> @One
		Begin
		/*********************************************************/
		/*				                 	 */
		/*********************************************************/
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
		Update	wyethaux.dbo.LTIDOC_OutstandingAwards
		SET	SAPFeed		= @UN
		Where	Opt_Num 	= @OptNum
		And	Destination	IN (@Trust,@SAP)

		SELECT 	@ERR = @@Error                   		IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

		/*********************************************************/
		/*				                 	 */
		/*********************************************************/
		End

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
Drop	Table	#OPTNUM

/*********************************************************************/
/*								     */
/*********************************************************************/
COMMIT TRAN	LTIDOCS
SET		@Tran	= @No

/*********************************************************************/
/*		OutPut Data 					     */
/*********************************************************************/
SELECT	@step = '30--Insert INTO #SAPFeed'

Insert	#SAPFeed
 (
	SAPFeed,TableName,[FileName],FileLoc,FileDest,FileLogName,FileErrorName,FileLog
 )

Select	SAPFeed, TableName,[FileName],FileLocation,FileDestination,FileLogName,FileErrorName,FileLog
From	LTIDOCSAPFeed
Where	SAP	= @Yes
And	Active	= @Yes

SELECT 	@ERR = @@Error
IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

/*********************************************************************/
/*								     */
/*********************************************************************/
Select	@iMax	= COUNT(SeqNo)
From	#SAPFeed

/*********************************************************************/
/*								     */
/*********************************************************************/
WHILE @iSeqNo <= @iMax
 Begin	---- BEGIN OUTER LOOP
	/*************************************************************/
	/*							     */
	/*************************************************************/
	SELECT	@step = '40--Insert INTO LTIDOCTempEndOfWyethAward & Dump Respective Out File(s)'

	Select	@SAPFeed 	= SAPFeed,
		@FileName	= FileLoc + [FileName],
		@ErrorName	= FileLog + FileErrorName,
		@LogName	= FileLog + FileLogName
	From	#SapFeed
	Where	SeqNo	= @iSeqNo

	/*************************************************************/
	/*	     Insert Detail Record(s) For BCP		     */
	/*************************************************************/
	Insert	WyethAux.dbo.LTIDOCTempEndOfWyethAward
	 (
		SAPFeed,OptNum,OptID,GrantNumber,PaidHyperion,PaidCurrencyCode,SAPFlag,RecordType,GID,LastName,InfoType,WageType,EffDate,Amount,BICode
	 )

	Select	LTI.SAPFeed,
		OPT.OPT_NUM,
		OPT.OPT_ID,
		LTI.GRANT_NUM,
		RTRIM(OPT.USER_CD2) As PAID_HYPERION,
		@USD As  PAID_CURRENCY_CODE,
		@Yes As SAPActive,
		@RType As RecordType,
		REPLACE(RTRIM(Address6),@GID,'') As GLOBAL_ID,
		OPT.NAME_LAST,
		@InfoType,
		PRO.SAP_WageType,
		SUBSTRING((CONVERT(Char(8),Getdate(),112)),5,4) + SUBSTRING((CONVERT(Char(8),Getdate(),112)),1,4) As EffeDate,	
		LTI.Dist_Dollars,
		PRO.SAP_BI_CODE

	From	wyethaux.dbo.LTIDOC_OutstandingAwards LTI INNER JOIN Wyeth.dbo.Optionee OPT
	ON	OPT.OPT_NUM = LTI.OPT_NUM
		INNER JOIN LTIDOCprocesses PRO
	ON	PRO.Process_ID = LTI.Process_ID
		INNER JOIN LTIDOCSAPFeed SAP
	ON	SAP.SAPFeed = LTI.SAPFeed

	Where	LTI.SAPFeed 	=  @SAPFeed
	And	SAP.SAP		=  @Yes
	And	SAP.Active	=  @Yes

	SELECT 	@ERR = @@Error,@RecordCount = @@ROWCOUNT
	IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

	/*************************************************************/
	/*							     */
	/*************************************************************/
	Insert	WyethAux.dbo.LTIDOCTempEndOfWyethAward
	 (
		SAPFeed,SAPFlag,RecordType,GID
	 )
	Select	@SAPFeed,
		@Yes,
		@TType,
		LTRIM(RTRIM(Convert(Char,@RecordCount)))

	SELECT 	@ERR = @@Error
	IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

	/*************************************************************/
	/*							     */
	/*************************************************************/
	Select	@SapCMD = 'Select RecordType,GID,LastName,WageType,EffDate,LTRIM(RTRIM(CONVERT(Char,Amount))),BICode From WyethAux.dbo.LTIDOCTempEndOfWyethAward Where SAPFeed = ' + @Tick + @SAPFeed + @Tick + 'Order By SeqNo'

	/*************************************************************/
	/*							     */
	/*************************************************************/
	Select @Command = 'Bcp "' + @SapCMD + '" queryout ' + @FileName  +  ' -S' + @@SERVERNAME + ' -T -c -e' + @ErrorName + ' ' + '>' + @LogName

	/*************************************************************/
	/*							     */
	/*************************************************************/
	EXEC @Err = Master..xp_cmdshell @Command, no_output

	SELECT 	@ERR = @@Error
	IF 	@ERR <> 0      GOTO   STEP99	--> --> --> --> --> --> --> --> --> --> --> --> --> 

	/*************************************************************/
	/*							     */
	/*************************************************************/
	Select	@iSeqNo	= @iSeqNo + @One

	/*************************************************************/
	/*							     */
	/*************************************************************/
 End	---- End OUTER LOOP

/*********************************************************************/
/*								     */
/*********************************************************************/
Drop	Table	#SAPFeed

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
SELECT	@FAIL		= @No              -- No errors thus far?  Set error-flag to NO...

/*********************************************************************/
STEP99:	--	<-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- <-- 
IF	@FAIL = @Yes
 Begin
	IF @Tran = @Yes
		ROLLBACK TRAN	LTIDOCS

	INSERT	WYETHaux.dbo.LTIDOCProcessLog
	VALUES 	(Getdate(), @Process,@ERR,'FAILED',@Process + ' FAILED AT STEP : '+ @step)
	RAISERROR (51112,16,1)   -- Custom SQL error-message in MASTER
	RETURN @ERR
 End
ELSE
 Begin
	IF @Tran = @Yes
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