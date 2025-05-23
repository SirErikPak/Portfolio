USE [WyethDBA]
GO
IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[P_Monitor_Process]') AND type in (N'P', N'PC'))
DROP PROCEDURE [dbo].[P_Monitor_Process]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE Proc [dbo].[P_Monitor_Process]
As
/*********************************************************************/
/*	   Trigger Monitor & Server Info Gather Process				     */
/*                  											     */
/* Purpose: Trigger Monitor & Server Info Gather Process             */
/*                                                                   */
/*  Org. Date: 01/01/2008        EPAK                                */
/*  Mod. Date: 00/00/0000        XXXX                                */
/*                                                                   */
/*********************************************************************/
SET NOCOUNT ON

/*********************************************************************/
/*																     */
/*********************************************************************/
Declare	@ErrorCode		Int,
		@ServerID		Int,
		@Process		Varchar(250),
		@procName		Varchar(128),
		@DateID			Int,
		@High			Char(4),
		@Zero			Int,
		@No				Char(1),
		@Yes			Char(1),
		@Out			Int,
		@JobDateInfo	Char(10),
		@ErrorFlag		Int,
		@ErrorMsg		Varchar(500),
		@Type			Char(1),
		@SpaceLimit		Int

/*********************************************************************/
/*																     */
/*********************************************************************/
Select	@Zero			= 0,
		@SpaceLimit		= 20,	---- Value in Percentage use to generate Space Warning
		@No 			= 'N',
		@Yes			= 'Y',
		@Process		= 'P_Monitor_Process',
		@ErrorFlag		= 0,
		@Out			= 1000,
		@High			= 'High',	---- Value Used for EMAIL Importance
		@JobDateInfo	= Convert(Char(10),Getdate(),121),
		@DateID			= MAX(DateID)
From	DateInfo

/*********************************************************************/
/*																     */
/*********************************************************************/
Select	@ServerID	= ServerID
From	ImportServerVersion
Where	ActiveFlag = @Yes

/*********************************************************************/
/*																     */
/*********************************************************************/
Insert ImportProcessLog
Values(@ServerID,Getdate(),@Process,@@Error, 'SUCCESS', '********** START Monitoring & Gather Server Info Processing **********')

/*********************************************************************/
/*					Clean Out Work Tables		                     */
/*********************************************************************/
Truncate Table	WorkDatabaseInfo
Truncate Table	WorkDiskInfo
Truncate Table	WorkScheduledJob
Truncate Table	WorkSQLUserInfo
Truncate Table	WorkDatabaseFile

/*********************************************************************/
/*																     */
/*********************************************************************/
IF (Select JobDateInfo From DateInfo Where DateID = (Select MAX(DateID) From DateInfo)) <> @JobDateInfo
OR (Select Count(DateID) From DateInfo) = @Zero	--- First Time Logic Only
 Begin
	/*************************************************************/
	/*				Insert New DateID Record					 */
	/*************************************************************/
	Insert DateInfo (JobDateInfo)
	Select Convert(Char(10),Getdate(),121)

		Select	@ErrorCode = @@ERROR
		IF @ErrorCode <> @Zero
		 Begin
			SET		@ErrorFlag = @Out
			SET		@ErrorMsg  = '1001 - Insert New DateInfo Table Failure'
			GOTO	Error_Trap
		 End

	/*************************************************************/
	/*				New DateID Record						     */
	/*************************************************************/
	Select	@DateID	= MAX(DateID)
	From	DateInfo

	Select	@JobDateInfo = JobDateInfo
	From	DateInfo
	Where	DateID = @DateID

 End
Else
 Begin
	/*************************************************************/
	/*														     */
	/*************************************************************/
	Update	DateInfo
	SET		Activeflag = @No
	Where	DateID = @DateID

		Select	@ErrorCode = @@ERROR
		IF @ErrorCode <> @Zero
		 Begin
			SET		@ErrorFlag = @Out
			SET		@ErrorMsg  = '1001 - Update ActiveFlag on DateInfo Table Failure'
			GOTO	Error_Trap
		 End

	Insert ImportProcessLog
	Values(@ServerID,Getdate(),@Process,@@ERROR, 'MESSAGE', 'Re-Run for DateID ' + (RTRIM(CONVERT(Varchar,@DateID))) + ' due to Incomplete or Possible Process Failure.')
 End

/*********************************************************************/
/*																     */
/*********************************************************************/
EXEC	@ErrorCode = P_Monitor_InsertEmailAddress	@ErrorCode,@Process,@ErrorFlag,@ErrorMsg OUT,
													@Yes,@No,@Zero,@Out,@ServerID

	IF @ErrorCode <> @Zero OR @@ERROR <> @Zero
	Begin
		SET @ErrorFlag	= @ErrorCode
		SET	@ProcName	= 'P_Monitor_InsertEmailAddress'		
		SET	@ErrorMsg	= '[' + RTRIM(ISNULL(@ErrorMsg,@ProcName)) + '} - Server ' + @@SERVERNAME + ' Gather Infomation Process Failure Occured...Please Check ' + @ProcName + ' Proc to determine Exact Point of Failure.'
		GOTO Error_Trap
	End

/*********************************************************************/
/*																     */
/*********************************************************************/
EXEC	@ErrorCode = P_Monitor_ServerScheduledJob	@DateID,@ErrorCode,@Process,@JobDateInfo,@ErrorFlag,
													@ErrorMsg OUT,@Yes,@No,@Zero,@Out,@ServerID


	IF @ErrorCode <> @Zero OR @@ERROR <> @Zero
	Begin
		SET @ErrorFlag	= @ErrorCode
		SET	@ProcName	= 'P_Monitor_ServerScheduledJob'		
		SET	@ErrorMsg	= '[' + RTRIM(ISNULL(@ErrorMsg,@ProcName)) + '} - Server ' + @@SERVERNAME + ' Gather Infomation Process Failure Occured...Please Check ' + @ProcName + ' Proc to determine Exact Point of Failure.'
		GOTO Error_Trap
	End

/*********************************************************************/
/*																     */
/*********************************************************************/
EXEC	@ErrorCode = P_Monitor_Combine	@DateID,@JobDateInfo,@Process,@ErrorCode,@ErrorFlag,
										@ErrorMsg OUT,@Yes,@No,@Zero,@Out,@ServerID

	IF @ErrorCode <> @Zero OR @@ERROR <> @Zero
	Begin
		SET @ErrorFlag	= @ErrorCode
		SET	@ProcName	= 'P_Monitor_Combine'
		SET	@ErrorMsg	= '[' + RTRIM(ISNULL(@ErrorMsg,@ProcName)) + '} - Server ' + @@SERVERNAME + ' Gather Infomation Process Failure Occured...Please Check ' + @ProcName + ' Proc to determine Exact Point of Failure.'
		GOTO Error_Trap
	End

/*********************************************************************/
/*																     */
/*********************************************************************/
EXEC	@ErrorCode = P_Monitor_Distribute	@DateID,@Process,@ErrorCode,@ErrorFlag,
											@ErrorMsg OUT,@Zero,@Yes,@Out,@ServerID

	IF @ErrorCode <> @Zero OR @@ERROR <> @Zero

	Begin
		SET @ErrorFlag	= @ErrorCode
		SET	@ProcName	= 'P_Monitor_Distribute'
		SET	@ErrorMsg	= '[' + RTRIM(ISNULL(@ErrorMsg,@ProcName)) + '} - Server ' + @@SERVERNAME + ' Gather Infomation Process Failure Occured...Please Check ' + @ProcName + ' Proc to determine Exact Point of Failure.'
		GOTO Error_Trap
	End

/*********************************************************************/
/*																     */
/*********************************************************************/
Update	DateInfo
Set		ActiveFlag = @Yes
Where	DateID = @DateID

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> @Zero
	 Begin
		SET		@ErrorFlag = @Out
		SET		@ErrorMsg  = '9990 - Unable to Set Active Flag on DateInfo Table'
		GOTO	Error_Trap
	 End

/*********************************************************************/
/*		Generate EMAIl For Unable TO Connect TO Link Server		     */
/*********************************************************************/
EXEC	@ErrorCode = P_Monitor_LinkServerMessage	@ErrorCode,@Process,@ErrorMsg OUT,@JobDateInfo,
													@ErrorFlag,@Out,@Zero,@Yes,@No,@High,@ServerID

	IF @ErrorCode <> @Zero OR @@ERROR <> @Zero
	Begin
		SET @ErrorFlag	= @ErrorCode
		SET	@ProcName	= 'P_Monitor_LinkServerErrorMessage'
		SET	@ErrorMsg	= '[' + RTRIM(ISNULL(@ErrorMsg,@ProcName)) + '} - Server ' + @@SERVERNAME + ' Gather Infomation Process Failure Occured...Please Check ' + @ProcName + ' Proc to determine Exact Point of Failure.'
		GOTO Error_Trap
	End

/********************************************************************/
/*																     */
/*********************************************************************/
EXEC	@ErrorCode = P_Monitor_ErrorWarningMessage	@ErrorCode,@ErrorFlag,@Process,@ErrorMsg OUT,@SpaceLimit,
													@JobDateInfo,@High,@Yes,@No,@Out,@Zero,@ServerID

	IF @ErrorCode <> @Zero OR @@ERROR <> @Zero
	Begin
		SET @ErrorFlag	= @ErrorCode
		SET	@ProcName	= 'P_Monitor_ErrorWarningMessage'
		SET	@ErrorMsg	= '[' + RTRIM(ISNULL(@ErrorMsg,@ProcName)) + '} - Server ' + @@SERVERNAME + ' Gather Infomation Process Failure Occured...Please Check ' + @ProcName + ' Proc to determine Exact Point of Failure.'
		GOTO Error_Trap
	End

/*********************************************************************/
/*																     */
/*********************************************************************/
Insert ImportProcessLog
Values(@ServerID,Getdate(),@Process,@@Error, 'SUCCESS', '********** END Monitoring & Gather Server Info Processing **********')

/*********************************************************************/
/*																     */
/*********************************************************************/
EXEC	@ErrorCode = P_Monitor_GatherProcessLog	@ErrorCode,@Process,@ErrorMsg OUT,@DateID,
												@JobDateInfo,@ErrorFlag,@Out,@Zero

	IF @ErrorCode <> @Zero OR @@ERROR <> @Zero
	Begin
		SET @ErrorFlag	= @ErrorCode
		SET	@ProcName	= 'P_Monitor_GatherProcessLog'
		SET	@ErrorMsg	= '[' + RTRIM(ISNULL(@ErrorMsg,@ProcName)) + '} - Server ' + @@SERVERNAME + ' Gather Infomation Process Failure Occured...Please Check ' + @ProcName + ' Proc to determine Exact Point of Failure.'
		GOTO Error_Trap
	End

/*********************************************************************/
/*																     */
/*********************************************************************/
Update	DatabaseServer
Set		ActiveFlag	= @Yes,
		ProcessFlag = @No
Where	ProcessFlag = @Yes

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> @Zero
	 Begin
		SET		@ErrorFlag = @Out
		SET		@ErrorMsg  = '9991 - Unable to Set Process Flag on DatabaseServer Table'
		GOTO	Error_Trap
	 End

/*********************************************************************/
/*																     */
/*********************************************************************/
Error_Trap:
IF @ErrorFlag <> @Zero
 Begin
	Insert ImportProcessLog
	Values(@ServerID,Getdate(),@Process,@ErrorFlag, 'FAILURE',@ErrorMsg)

/*********************************************************************/
/*				Send Failure EMAIL to All the Admins			     */
/*********************************************************************/
	EXEC	@ErrorCode = P_Monitor_ProcessFailureMessage	@ErrorCode,@Process,@ErrorMsg,@JobDateInfo,
															@ErrorFlag,@Out,@Zero,@Yes,@No,@High,@ServerID

/*********************************************************************/
/*																     */
/*********************************************************************/
	RAISERROR (90001,16,1)
	RETURN @ErrorFlag
 End

/*********************************************************************/
/*																     */
/*********************************************************************/
SET NOCOUNT OFF