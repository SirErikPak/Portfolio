USE [WyethDBA]
GO
IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[P_Monitor_ErrorWarningMessage]') AND type in (N'P', N'PC'))
DROP PROCEDURE [dbo].[P_Monitor_ErrorWarningMessage]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE   PROC [dbo].[P_Monitor_ErrorWarningMessage]	@ErrorCode		Int,
													@ErrorFlag		Int,
													@Process		Varchar(250),
													@ErrorMsg		Varchar(500) OUTPUT,
													@SpaceLimit		Int,
													@JobDateInfo	Char(10),
													@High			Char(4),
													@Yes			Char(1),
													@No				Char(1),
													@Out			Int,
													@Zero			Int,
													@ServerID		Int
As
/*********************************************************************/
/* Database: WyethDBA					  						     */
/* Purpose: Prep EMAIL for Any Failure(s) & Disk Storage Check	     */
/* Note: 															 */
/*                                                                   */
/*  Org. Date: 09/06/2008        EPAK                                */
/*  Mod. Date: 00/00/0000                                            */
/*********************************************************************/
SET NOCOUNT ON

/*********************************************************************/
/*              Declare Variables                                    */
/*********************************************************************/
Declare		@Max			Int,			
			@SeqNo			Int,
			@SiteID			Int,
			@One			Int,
			@Two			Int,
			@Three			Int,
			@Counter		Int,
			@MsgID			Int,
			@Cdrive			Char(1),
			@Tick			Char(1),
			@DBSpaceLimit	Float,
			@DBName			Varchar(128),
			@OwnerName		Varchar(128),
			@Subject		Varchar(250),
			@Message		Varchar(8000),
			@iMessage		Varchar(8000),
			@iiMessage		Varchar(8000),
			@iiiMessage		Varchar(8000),
			@Sql			Varchar(8000)
			
/*********************************************************************/
/*													                 */
/*********************************************************************/
SET	@Process		= @Process + ' - [P_Monitor_ErrorWarningMessage]'
SET	@Cdrive			= 'C'				 ---- By Pass "C" Dirve for Space Check
SET	@DBName			= 'WyethDBA'
SET	@OwnerName		= 'dbo'
SET	@Tick			= ''''
SET	@DBSpaceLimit	= @SpaceLimit/100.00 ----- Converting Data INTO Decimal Format

/*********************************************************************/
/*													                 */
/*********************************************************************/
Select	@One		= 1,
		@Two		= 2,
		@Three		= 3,
		@Counter	= 0,
		@Out		= 7000,
		@SeqNo		= 1,
		@Message	= 'Please review the attachment for detail information about MS SQL Server Schecdule Jobs Failure(s). **** Review WyethDBA.dbo.ProcessLog For Additional Infomation ****',
		@iMessage	= 'Please review the attachment for detail information about Server Space Warning(s).  **** Review WyethDBA.dbo.ProcessLog For Additional Infomation ****',
		@iiMessage	= 'Please review the attachment for detail information about Server(s) with No or Old Server/Database Statistics.  **** Review WyethDBA.dbo.ProcessLog For Additional Infomation ****',
		@iiiMessage	= 'Please review the attachment for detail information about Database(s) Space WARNING.  **** Review WyethDBA.dbo.DatabaseFileSummary And/Or WyethDBA.dbo.DatabaseFile View For Additional Infomation ****'

/*********************************************************************/
/*													                 */
/*********************************************************************/
Create Table #Site
(
	SeqNo	Int	Identity,
	SiteID	Int	NOT NULL,
	MsgID	Int	NOT NULL
)

	Select	@ErrorCode = @@ERROR
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorFlag = @Out
		SET @ErrorMsg  = '7001 - Temp Table #Site Create Failure'
		GOTO Error_Flag
	 End

/*********************************************************************/
/*  	      Populate Site Table for Failure EMAIL                  */
/*********************************************************************/
Insert	#Site (SiteID,MsgId)
Select	DISTINCT SiteID,@Zero
From	vw_FailScheduledJob

	Select	@ErrorCode = @@ERROR
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorFlag = @Out
		SET @ErrorMsg  = '7002 - Insert Into #Site Temp Table for MS SQL Scheduled Job Failure'
		GOTO Error_Flag
	 End

/*********************************************************************/
/* 	      Populate Site Table for Space Warning EMAIL                */
/*********************************************************************/
Insert	#Site (SiteID,MsgId)
Select	DISTINCT SiteID, @One
From	Vw_ServerStorage
Where	DriveLetter <> @Cdrive
And		ServerID NOT IN (Select ServerID From DatabaseServerDiskExclude)

	Select	@ErrorCode = @@ERROR
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorFlag = @Out
		SET @ErrorMsg  = '7002 - Insert Into #Site Temp Table for Space Warning'
		GOTO Error_Flag
	 End

/*********************************************************************/
/*  Populate Site Table for Missing Record(s) From Import Table(s)   */
/*********************************************************************/
Insert	#Site (SiteID,MsgId)
Select	DISTINCT SER.SiteID,@Two
From	ImportProcessLog PLOG INNER JOIN DatabaseServer SER
ON		PLOG.ServerID = SER.ServerID
Where	PLOG.SuccessFailure = 'WARNING'
And		PLOG.ErrorCode IN (900)
And		SER.ActiveFlag = @Yes

	Select	@ErrorCode = @@ERROR
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorFlag = @Out
		SET @ErrorMsg  = '7002 - Insert Into #Site Temp Table for Empty or OLD Process Info'
		GOTO Error_Flag
	 End

/*********************************************************************/
/*  Populate Site Table for Missing Record(s) From Import Table(s)   */
/*********************************************************************/
Insert	#Site (SiteID,MsgId)
Select	DISTINCT CONVERT(Int,SiteID),@Three
From	vw_DatabaseFileSummary
Where	CONVERT(DECIMAL(18,7),PercentFree) < @DBSpaceLimit

	Select	@ErrorCode = @@ERROR
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorFlag = @Out
		SET @ErrorMsg  = '7003 - Insert Into #Site Temp Table for Database Space Warning'
		GOTO Error_Flag
	 End

/*********************************************************************/
/*													                 */
/*********************************************************************/
Select	@Max	= Count(SeqNo)
From	#Site

/*********************************************************************/
/*													                 */
/*********************************************************************/
WHILE @SeqNo <= @Max
 Begin
	/*****************************************************************/
	/*												                 */
	/*****************************************************************/
	Select	@SiteID = SiteID,
			@MsgID	= MsgID
	From	#Site
	Where	SeqNo = @SeqNo

	IF @MsgID = @Zero
	BEGIN
	/*****************************************************************/
	/*			Create SQL Statement for Each Site	                 */
	/*****************************************************************/
	SET @Sql = 'Select FailureInfo From ' + @DBName + '.' + @Ownername + '.Vw_FailScheduledJob Where SiteID = ' + RTRIM(CONVERT(Char,@SiteID))
	SET @Subject = 'FAILURE(s) - MS SQL Server Scheduled Job(s)'

	/*************************************************************/
	/*   Generate Message for Each Site for Scheduled Jobs       */
	/*************************************************************/
	Insert	DatabaseEmailMessage
	(
		SiteID,EMailAddress,CC,Priority,EMailSubject,EMailMsg,Query
	)

	Select	@SiteID,
			EmailTo,
			EmailCC,
			@High,
			@Subject,
			@Message,
			@Sql
	From	DatabaseEmail
	Where	SiteID = @SiteID
	
		Select	@ErrorCode = @@ERROR
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorFlag = @Out
			SET @ErrorMsg  = '7003 - Insert Into DatabaseEmailMessage Table Failure'
			GOTO Error_Flag
		 End

	END

	IF @MsgID = @One
	BEGIN
	/*****************************************************************/
	/*			Create SQL Statement for Each Site	                 */
	/*****************************************************************/	
	SET @Sql = 'Select + '' ['' + ServerTypeDesc +  + ''] '' + ServerName + '' Server at '' + SiteDesc + '' - As of ' + @JobDateInfo + ' Drive ['' + DriveLetter + ''] Has Less Than Equal to '' + RTRIM(CONVERT(Char,FreeDiskPercentage)) + '' Percent Free Space ['' + ' + 'LTRIM(RTRIM(FreeDiskSpace_MB)) + '' MB Free] '' From ' + RTRIM(@DBName) + '.' + RTRIM(@OwnerName) + '.Vw_ServerStorage Where SiteID = ' + RTRIM(CONVERT(Char,@SiteID)) + ' And ServerID NOT IN (Select ServerID From ' + RTRIM(@DBName) + '.' + RTRIM(@OwnerName) + '.DatabaseServerDiskExclude) And DriveLetter <> ''' + @CDrive + ''''
	SET @Subject = 'WARNING(s) - MS SQL Server Disk Space Message'

	/*************************************************************/
	/*   Generate Message for Each Site for Server Space         */
	/*************************************************************/
	Insert	DatabaseEmailMessage
	(
		SiteID,EMailAddress,Priority,EMailSubject,EMailMsg,Query
	)

	Select	@SiteID,
			EmailTo,
			@High,
			@Subject,
			@iMessage,
			@Sql
	From	DatabaseEmail
	Where	SiteID = @SiteID
	
		Select	@ErrorCode = @@ERROR
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorFlag = @Out
			SET @ErrorMsg  = '7004 - Insert Into DatabaseEmailMessage Table Failure'
			GOTO Error_Flag
		 End
	END

	/*****************************************************************/
	/*												                 */
	/*****************************************************************/
	IF @MsgID = @Two
	BEGIN
	/*****************************************************************/
	/*			Create SQL Statement for Each Site	                 */
	/*****************************************************************/	
	SET @Sql = 'Select + '' ['' + DT.ServerTypeDesc + ''] '' + S.ServerName + '' Server At '' + ST.SiteDesc + '' - '' + [Message] From ' + RTRIM(@DBName) + '.' + RTRIM(@OwnerName) +  '.ImportProcessLog P LEFT JOIN ' + RTRIM(@DBName) + '.' + RTRIM(@OwnerName) + '.DatabaseServer S ON S.ServerID=P.ServerID INNER JOIN ' + RTRIM(@DBName) + '.' + RTRIM(@OwnerName) + '.DatabaseSite ST ON ST.SiteID = S.SiteID INNER JOIN ' + RTRIM(@DBName) + '.' + RTRIM(@OwnerName) +  '.DatabaseServerType DT ON S.ServerTypeID = DT.ServerTypeID Where P.SuccessFailure = ''WARNING'' And P.ErrorCode IN (900) And ST.SiteID = ' + RTRIM(CONVERT(Char,@SiteID))
	SET	@Subject = 'WARNING - No Data to Retrive or OLD Data from the Remote Server(s)'

	/*************************************************************/
	/*   Generate Message for Each Site for Server Space         */
	/*************************************************************/
	Insert	DatabaseEmailMessage
	(
		SiteID,EMailAddress,Priority,EMailSubject,EMailMsg,Query
	)

	Select	@SiteID,
			EmailTo,
			@High,
			@Subject,
			@iiMessage,
			@Sql
	From	DatabaseEmail
	Where	SiteID = @SiteID
	
		Select	@ErrorCode = @@ERROR
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorFlag = @Out
			SET @ErrorMsg  = '7005 - Insert Into DatabaseEmailMessage Table Failure'
			GOTO Error_Flag
		 End

	END

	/*****************************************************************/
	/*												                 */
	/*****************************************************************/
	IF @MsgID = @Three
	BEGIN
	/*****************************************************************/
	/*			Create SQL Statement for Each Site	                 */
	/*****************************************************************/
	SET @Sql = 'Select	 + '' ['' + ServerTypeDesc + '']  Database ['' + DatabaseName + ''] & File Type ['' + FileType + ''] on Server ['' + ServerName + ''] Located at ['' + SiteDescription + ''] has '' + LTRIM(RTRIM(CONVERT(Char,CONVERT(Money,(PercentFree*100)),1))) + ''% free Space '' From ' + RTRIM(@DBName) + '.' + RTRIM(@OwnerName) + '.vw_DatabaseFileSummary Where PercentFree < ' + LTRIM(RTRIM(CONVERT(Char,@DBSpaceLimit))) + ' And SiteID = ' + RTRIM(CONVERT(Char,@SiteID))
	SET	@Subject = 'WARNING - Database Space is UNDER ' + RTRIM(CONVERT(Char,@Spacelimit)) + '%'

	/*************************************************************/
	/*   Generate Message for Each Site for Server Space         */
	/*************************************************************/
	Insert	DatabaseEmailMessage
	(
		SiteID,EMailAddress,Priority,EMailSubject,EMailMsg,Query
	)

	Select	@SiteID,
			EmailTo,
			@High,
			@Subject,
			@iiiMessage,
			@Sql
	From	DatabaseEmail
	Where	SiteID = @SiteID
	
		Select	@ErrorCode = @@ERROR
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorFlag = @Out
			SET @ErrorMsg  = '7005 - Insert Into DatabaseEmailMessage Table Failure'
			GOTO Error_Flag
		 End

	END

	/*****************************************************************/
	/*												                 */
	/*****************************************************************/
	Select	@SeqNo = @SeqNo + @One

	/*****************************************************************/
	/*												                 */
	/*****************************************************************/
 End

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
DROP TABLE #Site

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
Insert ImportProcessLog
Values(@ServerID,Getdate(),@Process, @@ERROR, 'SUCCESS','***** COMPLETED - Report Any Scheuled Job Failures & Server Storage Checks')

/*********************************************************************/
/*																     */
/*********************************************************************/
Error_Flag:
IF @ErrorFlag <> @Zero
 Begin
	Insert ImportProcessLog
	Values(@ServerID,Getdate(),@Process,@ErrorFlag, 'FAILURE',@ErrorMsg)
	RETURN @ErrorFlag
 End

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT OFF