if exists (select * from dbo.sysobjects where id = object_id(N'[dbo].[P_GECS_PrimaryRecord_BCP]') and OBJECTPROPERTY(id, N'IsProcedure') = 1)
drop procedure [dbo].[P_GECS_PrimaryRecord_BCP]
GO
CREATE Proc P_GECS_PrimaryRecord_BCP
As
/*********************************************************************/
/*		GEAC - BCP					     */
/*                  						     */
/* Purpose: Determine Primary Record for GEAC		             */
/*                                                                   */
/*  Org. Date: 08/03/2009        EPAK                                */
/*  Mod. Date: 00/00/0000        XXXX                                */
/*                                                                   */
/*********************************************************************/
SET NOCOUNT ON

/*********************************************************************/
/*								     */
/*********************************************************************/
DECLARE		@Yes		Char(1),
		@No		Char(1),
		@Error		Char(1),
		@Step		Varchar(256),
		@SeqNo		Int,
		@Max		Int,
		@One		Int,
		@Zero		Int,
		@ErrorCode	Int,
		@Process	Varchar(256),
		@Out		Char(1),
		@In		Char(1),
		@BcpFlag	Char(1),
		@BackupFlag	Char(1),
		@CopyFlag	Char(1),
		@YYYYMMDD	Char(8),
		@YYYYMM		Char(8),
		@INP		Char(4),
		@OutIn		Varchar(5),
		@TableName	Varchar(255),
		@FileName	Varchar(255),
		@FmtName	Varchar(255),
		@Command	Varchar(8000),
		@ErrorName	Varchar(255),
		@LogName	Varchar(255),
		@FileSource	Varchar(255),
		@FileTarget	Varchar(255),
		@BackupLocation Varchar(255)

/*********************************************************************/
/*		Assign Local Variables				     */
/*********************************************************************/
Select		@Yes		= 'Y',
		@No		= 'N',
		@SeqNo		= 1,
		@One		= 1,
		@Zero		= 0,
		@Error		= 'Y',
		@INP		= '.inp',
		@Process	= 'P_GECS_PrimaryRecord_BCP'

Select		@Out		= 'O',
		@In		= 'I',
		@YYYYMMDD	= (CONVERT(Char(8),Getdate(),112)),
		@YYYYMM		= '_' + SUBSTRING(CONVERT(Char,(DATEADD(Month,-2,Getdate())),112),1,6) + '*'

/*********************************************************************/
/*								     */
/*********************************************************************/
SET	@Step	= '01 - Create Temp #BCP Table'

Create	Table	#BCP
 (
	SeqNo			Int		Identity,
	FileSourceFolder	Varchar(255)	NOT NULL,
	FileTargetFolder	Varchar(255)	NOT NULL,
	[FileName]		Varchar(128)	NOT NULL,
	FileExt			Char(4)		NOT NULL,
	TableName		Varchar(255)	NOT NULL,
	LogName			Varchar(255)	NOT NULL,
	ErrorFileName		Varchar(255)	NOT NULL,
	BackupLocation		Varchar(255)	NOT NULL,
	FmtFileLocation		Varchar(255)	NOT NULL,
	FmtName			Varchar(255)	NOT NULL,
	FmtExt			Char(4)		NOT NULL,
	OutInFlag		Char(1)		NOT NULL,
	BackupFlag		Char(1)		NOT NULL,
	CopyFlag		Char(1)		NOT NULL
 )


	SELECT 	@ErrorCode = @@ERROR		IF @ErrorCode <> @Zero
			GOTO   SucessError


Create	Table	#RecordCount
 (
	SeqNo			Int		Identity,
	RecordCount		Int		NOT NULL
 )


	SELECT 	@ErrorCode = @@ERROR		IF @ErrorCode <> @Zero
			GOTO   SucessError

/*********************************************************************/
/*								     */
/*********************************************************************/
SET	@Step	= '02 - Inserte BCP Information into #BCP Table'

Insert	#BCP
 (
	FileSourceFolder,FileTargetFolder,[FileName],FileExt,TableName,LogName,ErrorFileName,BackupLocation,FmtFileLocation,FmtName,FmtExt,OutInFlag,BackupFlag,CopyFlag
 )
Select	FileSourceFolder,
	FileTargetFolder,
	[FileName],
	FileExtension,
	TableName,
	LogName,
	ErrorFileName,
	BackupLocation,
	FmtFileLocation,
	FmtFileName,
	FmtExtension,
	OutInFlag,
	BackupFlag,
	CopyFlag
From 	BCPInformation
Where	ActiveFlag = @Yes


	SELECT 	@ErrorCode = @@ERROR		IF @ErrorCode <> @Zero
			GOTO   SucessError
/*********************************************************************/
/*								     */
/*********************************************************************/
Select	@Max	= Count(*)	
From	#BCP

/*********************************************************************/
/*								     */
/*********************************************************************/
WHILE @SeqNo <= @Max
 Begin	---- BEGIN LOOP
	/*************************************************************/
	/*							     */
	/*************************************************************/
	Select	@FileSource	= FileSourceFolder,
		@FileTarget	= FileTargetFolder,
		@FileName	= FileSourceFolder + [FileName] + FileExt,
		@TableName	= TableName,
		@LogName	= LogName,
		@ErrorName	= ErrorFileName,
		@BackupLocation	= BackupLocation,
		@FmtName	= FmtFileLocation + FmtName + FmtExt,
		@BcpFlag	= OutInFlag,
		@BackupFlag	= BackupFlag,
		@CopyFlag	= CopyFlag
	From	#BCP
	Where	SeqNo = @SeqNo

	/*************************************************************/
	/*							     */
	/*************************************************************/
	SET	@Step	= '50 - BCP Failure'

	IF @BCPFlag = @Out
	 Begin
		Select	@OutIn = ' Out '
		Select @Command = 'Bcp ' + @TableName + @OutIn + @FileName  +  ' -S' + @@SERVERNAME + '  -T -f' + @FMTName + ' -e' + @ErrorName + ' ' + '>' + @LogName
	 End
--	ELSE IF @BCPFlag = @In
--	 Begin
--		Select	@OutIn = ' In '
--		Select @Command = 'Bcp ' + @TableName + @OutIn + @FileName + ' -S' + @@SERVERNAME + ' -T -f' + @FMTName + ' -e' + @ErrorName + ' ' + '>' + @LogName
--	 End
	ELSE
	 Begin
		SELECT 	@ErrorCode = -999			GOTO   SucessError
	 End

	/*************************************************************/
	/*							     */
	/*************************************************************/
	EXEC @ErrorCode = Master..xp_cmdshell @Command, no_output


			IF @ErrorCode <> @Zero
				GOTO   SucessError

	/*************************************************************/
	/*							     */
	/*************************************************************/
	SET	@Step	= '51 - Insert #RecordCount & Recount Info Failure'

	Select	@Command = 'Select Count(*) From ' + RTRIM(TableName)
	From	#BCP
	Where	SeqNo = @SeqNo


	Insert	#RecordCount
	Exec 	(@command)


		SELECT 	@ErrorCode = @@ERROR			IF @ErrorCode <> @Zero
				GOTO   SucessError

	/*************************************************************/
	/*          BackUp Command Build String                      */
	/*************************************************************/
	IF @BackupFlag = @Yes

	 Begin
		/*****************************************************/
		/*						     */
		/*****************************************************/
		SET	@Step	= '51 - Backup Failure'

		Select	@Command = 'Copy '  + @FileSource  + [FileName] + FileExt
			+ ' '  + @BackupLocation + [FileName] + '_' + @YYYYMMDD + FileExt
		From	#BCP
		Where	SeqNo = @SeqNo

		/*****************************************************/
		/*						     */
		/*****************************************************/
		EXEC @ErrorCode = Master..xp_cmdshell @Command, no_output


				IF @ErrorCode <> @Zero
					GOTO   SucessError

		/*****************************************************/
		/*						     */
		/*****************************************************/
	 End

	/*************************************************************/
	/*            Copy Command Build String                      */
	/*************************************************************/
	IF @CopyFlag = @Yes
	 Begin
		/*****************************************************/
		/*						     */
		/*****************************************************/
		SET	@Step	= '52 - Copy Failure'

		Select	@Command = 'Copy '  + @FileSource + [FileName] + FileExt
			+ ' '  + @FileTarget + [FileName] + @INP
		From	#BCP
		Where	SeqNo = @SeqNo

		/*****************************************************/
		/*						     */
		/*****************************************************/
		EXEC @ErrorCode = Master..xp_cmdshell @Command, no_output

				IF @ErrorCode <> @Zero
					GOTO   SucessError

		/*****************************************************/
		/*						     */
		/*****************************************************/
		SET	@Step	= '53 - Rename Copy Failure'

		Select	@Command = 'Ren '  + @FileTarget + [FileName] + @INP
			 + ' ' + [FileName] + FileExt
		From	#BCP
		Where	SeqNo = @SeqNo

		/*****************************************************/
		/*						     */
		/*****************************************************/
		EXEC @ErrorCode = Master..xp_cmdshell @Command, no_output
				IF @ErrorCode <> @Zero
					GOTO   SucessError

		/*****************************************************/
		/*						     */
		/*****************************************************/
	 End

	/*************************************************************/
	/*            	   Delete Empty File(s)                      */
	/*************************************************************/
	IF	(Select RecordCount From #RecordCount Where SeqNo = @SeqNo)  = @One
		OR (Select RecordCount From #RecordCount Where SeqNo = @SeqNo)  = @Zero
	 Begin
		/*****************************************************/
		/*		Delete BCP File			     */
		/*****************************************************/
		SET	@Step	= '54 - Delete BCP File Failure'

		Select	@Command = 'Del '  + @FileSource + [FileName] + FileExt
		From	#BCP
		Where	SeqNo = @SeqNo

		/*****************************************************/
		/*						     */
		/*****************************************************/
		EXEC @ErrorCode = Master..xp_cmdshell @Command, no_output


				IF @ErrorCode <> @Zero
					GOTO   SucessError

		/*****************************************************/
		/*						     */
		/*****************************************************/		
	 End
		
	/*************************************************************/
	/*	Delete OLD Backup Greater Than 60 Days Old	     */
	/*************************************************************/
	SET	@Step	= '55 - Delete OLD Backup Failure'

	Select	@Command = 'Del '  + @BackupLocation + [FileName] + @YYYYMM + FileExt
	From	#BCP
	Where	SeqNo = @SeqNo

	/*************************************************************/
	/*							     */
	/*************************************************************/
	EXEC @ErrorCode = Master..xp_cmdshell @Command, no_output

	/*************************************************************/
	/*							     */
	/*************************************************************/
	Select	@SeqNo = @SeqNo + @One
	
	/*************************************************************/
	/*							     */
	/*************************************************************/
 End	---- End LOOP

/*********************************************************************/
/*		House Cleaning					     */
/*********************************************************************/
DROP Table	#BCP
SET	@Error = @No

/*********************************************************************/
/*								     */
/*********************************************************************/
SucessError:
IF	@Error = @Yes
 Begin
	INSERT	PfizerInterface.dbo.ProcessLog
	VALUES 	(Getdate(), @Process,@ErrorCode,'FAILURE',@Process + ' FAILED AT STEP : '+ @Step)
	RAISERROR (55002,16,1)   -- Custom SQL error-message in MASTER
	RETURN	@ErrorCode

 End
ELSE
 Begin
	INSERT 	PfizerInterface.dbo.ProcessLog
	VALUES	(Getdate(), @Process,@ErrorCode,'SUCCESS',@Process + ' Completed')

 End

/*********************************************************************/
/*								     */
/*********************************************************************/
SET NOCOUNT OFF